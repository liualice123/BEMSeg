import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tsnecuda import TSNE
from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor
from itertools import chain
from .untils import tokenize
from .Draw_TSNE import plotlabels


@SEGMENTORS.register_module()
class DenseCLIP(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 text_encoder,
                 context_decoder,
                 decode_head,
                 class_names,
                 context_length,
                 context_feature='attention',
                 score_concat_index=3,
                 text_head=False,
                 neck=None,
                 tsne=False,
                 tau=0.07,
                 auxiliary_head=None,
                 identity_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 # texts_simple=None,
                 token_embed_dim=512, text_dim=1024,
                 **args):
        super(DenseCLIP, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained

            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'
            
            if 'RN50' not in pretrained and 'RN101' not in pretrained and 'ViT-B' not in pretrained:
                print('not CLIP pre-trained weight, using CLIP ViT-B-16')
                text_encoder.pretrained = 'pretrained/ViT-B-16.pt'
            else:
                text_encoder.pretrained = pretrained

        # self.class_names =
        self.backbone = builder.build_backbone(backbone)
        self.text_encoder = builder.build_backbone(text_encoder)
        self.context_decoder = builder.build_backbone(context_decoder)
        self.context_length = context_length      # 5
        self.score_concat_index = score_concat_index

        assert context_feature in ['attention', 'backbone']
        self.context_feature = context_feature

        self.text_head = text_head
        self.tau = tau
        self.tsne = tsne

        if neck is not None:
            self.neck = builder.build_neck(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.with_identity_head = False
        self.identity_head = None
        self._init_identity_head(identity_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.texts_simple = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])


        self.num_classes = len(self.texts)


        context_length = self.text_encoder.context_length - self.context_length
        self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)
        self.gamma2 = nn.Parameter(torch.ones(text_dim) * 1e-4)

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)
    
    def _init_identity_head(self, identity_head):
        """Initialize ``auxiliary_head``"""
        if identity_head is not None:
            self.with_identity_head = True
            self.identity_head = builder.build_head(identity_head)

    def extract_feat(self, img):
        """Extract features from images."""
        # print(len(img))
        # print(img[0].shape)
        x = self.backbone(img)
        return x

    def before_extract_feat(self, img):
        B, C, H, W = img.shape
        image_embeddings = F.normalize(img, dim=1, p=2)
        text_embeddings = self.text_encoder(self.texts.to(img.device), self.contexts).expand(B, -1, -1)
        # print("test")
        # print(text_embeddings.size())  # (4,6,1024)
        text_em = text_embeddings[:, :, :3]
        text_ = F.normalize(text_em.to(img.device), dim=2, p=2)
        class_map = torch.einsum('bchw,bkc->bkhw', image_embeddings, text_)
        img = torch.cat([img, class_map], dim = 1)
        return img

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses
    def _decode_head_forward(self, x):
        """Run forward function and calculate loss for decode head in
        training."""

        decode_out = self.decode_head.forward(x)
        return decode_out

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def _identity_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        loss_aux = self.identity_head.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)

        losses.update(add_prefix(loss_aux, 'aux_identity'))
        # losses.update(add_prefix(loss_aux1, 'aux_identity1'))
        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def after_extract_feat(self, x):
        # no multimodal feature fusion
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]
        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        # (B, K, C)
        text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)

        # DenseCLIP 加上下面两行
        # text_diff = self.context_decoder(text_embeddings, visual_context)
        # text_embeddings = text_embeddings + self.gamma * text_diff
        # compute score map and concat
        _, K, _ = text_embeddings.shape
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
        x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)
        return text_embeddings, x_orig, score_map

    # 只返回文本特征用于作图，包括多模态特征融合之前的和之后的
    def extract_text_feat(self, x):
        # no multimodal feature fusion
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]
        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        # (B, K, C) 融合之前
        text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        # 融合之后
        # text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        # text_diff = self.context_decoder(text_embeddings, visual_context)
        # text_embeddings = text_embeddings + self.gamma * text_diff

        # compute score map and concat
        B, K, C = text_embeddings.shape
        text = F.normalize(text_embeddings, dim=2, p=2)
        print(B,K,C)
        return text

    def new_after_extract_feat(self, x):
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]
        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        visual_one = visual_embeddings.reshape(B, C, H*W).permute(0, 2, 1)
        # (B, K, C)
        text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)   # -1 表示取当前维度不变，因为编码的文本向量是没有batch的，所以给他加上了B

        # update text_embeddings by visual_context!
        # (B, 1, C)
        vision_diff = self.context_decoder(visual_one, text_embeddings)
        vision_diff = self.gamma2 * vision_diff

        text_diff = self.context_decoder(text_embeddings, visual_context)

        # print("--------------------test------------------")
        # print(text_diff.shape)
        # (B, K, C)
        text_embeddings = text_embeddings + self.gamma * text_diff
        # vision_diff = vision_diff.permute(0, 2, 1).reshape(B, C, H, W)
        visual_embeddings = visual_embeddings + vision_diff.permute(0, 2, 1).reshape(B, C, H, W)

        # compute score map and concat
        B, K, C = text_embeddings.shape
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
        x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)
        return text_embeddings, x_orig, score_map

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)
        global_, visual_ = x[4]
        _x_orig = [x[i] for i in range(4)]

        text_embeddings, x_orig, score_map = self.after_extract_feat(x)
        # text_feature = self.extract_text_feat(x)

        if self.with_neck:
            x_orig = list(self.neck(x_orig))
            _x_orig = x_orig

        losses = dict()
        if self.text_head:
            x = [text_embeddings,] + x_orig
        else:
            x = x_orig

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        # print("-------------------test-------------------")
        # print("设置断点")
        # print(type(gt_semantic_seg))
        # print(torch.max(gt_semantic_seg))     # JLYH是【8，1，512，512】而且值比较多，255，34，35，39，40，43，44，45，58，66，76，80，91，95，96，
        # print(torch.min(gt_semantic_seg))
        # print(gt_semantic_seg[0])    # potsdam也是【8，1，512，512】，但是对应的值只有0，1，2，3，4，5，255，也就是说需要先对JLYH的标签值进行调整
        # t = numpy.array(gt_semantic_seg)
        # print(numpy.max(gt_semantic_seg[0]))
        # print(max(gt_semantic_seg))

        losses.update(loss_decode)

        if self.with_identity_head:     #  true
            loss_identity = self._identity_head_forward_train(
                score_map/self.tau, img_metas, gt_semantic_seg)
            losses.update(loss_identity)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                _x_orig, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        # 先保存特征，然后用T-SNE展示特征
        show_plot = False
        score = False
        text = False
        with torch.no_grad():
            if show_plot:
                feature = self._decode_head_forward(x)  # 128的大小
                seg_logit = resize(
                    input=feature,
                    size=gt_semantic_seg.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
                # print(seg_logit.shape)   # 512的大小
                B, C, H, W = seg_logit.shape

                feature = seg_logit.permute(0, 2, 3, 1)   # B,H,W,C
                print("---------------------输出大小")

                feature = feature.reshape(-1, C)
                print(feature.shape)
                gt_label = gt_semantic_seg.permute(0, 2, 3, 1).reshape(-1, 1) #B, H,W,C
                gt_label [gt_label==255] = 6
                print(gt_label.shape)
                # print(torch.unique(gt_label))
                feature = numpy.array(feature.cpu())
                gt_label = numpy.array(gt_label.cpu())
                # gt_label_2 = numpy.array(list(chain.from_iterable(gt_label)))

                # feature = feature.tolist()
                # gt_label = list(gt_label)

                X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(feature[:10000])
                print("特征先压缩")
                #
                plotlabels(X_embedded, gt_label[:10000], 'title')

                print("有没有结果")
# 先保存好特征，再用T-SNE进行展示，因为整个训练流程是在cuda上进行的,输出的有可能是中间省略的，用保存到文件的方式来记录。
#             print(label)
#             print(gt_label)
#             filename = 'write_data3.txt'
#             with open(filename, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
#                 for data in feature:
#                     f.write(str(data)+"\n")
#                 f.write(str(gt_label))
            # draw_tsne(feature, gt_label)

        if score:
            # 先来看特征，x[4]是全局特征和局部特征

            B, C1, H, W = visual_.size()
            visual_TNE = visual_.permute(0, 2, 3, 1).reshape(-1, C1)
            print(visual_TNE.size())
            visual_TNE = visual_TNE.tolist()

            #再来看标签 score map是 bchw
            score_pro = F.softmax(score_map / self.tau, 1)
            B,C,H,W = score_pro.size()
            print(C)
            sco = score_pro.permute(0, 2, 3, 1).reshape(-1, C)
            lab = torch.argmax(sco, 1)
            # print(lab.size())
            label = numpy.array(lab.cpu())
            # gt_label_2 = numpy.array(list(chain.from_iterable(gt_label)))

            label = list(label)

            # 最后把它们写入文件
            filename = 'write_data2.txt'
            with open(filename, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
                for data in visual_TNE:
                    f.write(str(data) + "\n")
                f.write(str(label))
                print(label)

        if text:
            B, N, C = text_feature.size()
            text_feature = text_feature.reshape(-1, C)
            print(text_feature.size())
            text_TNE = text_feature.tolist()
            # 最后把它们写入文件
            label = []
            for j in range(0, B):
                for i in range(0, N):
                    label.append(i)
            filename = 'text_data.txt'
            with open(filename, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
                for data in text_TNE:
                    f.write(str(data) + "\n")

                f.write(str(label))
                print(label)

        # print("--------------over------------------")
        return losses

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # img = self.before_extract_feat(img)
        x = self.extract_feat(img)

        _x_orig = [x[i] for i in range(4)]
        text_embeddings, x_orig, score_map = self.after_extract_feat(x)

        if self.with_neck:
            x_orig = list(self.neck(x_orig))

        if self.text_head:
            x = [text_embeddings,] + x_orig
        else:
            x = x_orig
        # print('text_embedding=', text_embeddings[0])
        out = self._decode_head_forward_test(x, img_metas)
        # print('cls_map=', out[0,:,40, 40])
        # if self.tsne:
        #     x_tsne = x[1]
        #     print(x_tsne.shape)
        #     out = resize(
        #         input=x_tsne,
        #         size=img.shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        # else:
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out


    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        
        if  torch.isnan(seg_logit).any():
            print('########### find NAN #############')

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        if self.tsne:
            seg_logit = self.encode_decode(img, img_meta)
            seg_pred = seg_logit
        else:

            seg_logit = self.inference(img, img_meta, rescale)
            seg_pred = seg_logit.argmax(dim=1)
            if torch.onnx.is_in_onnx_export():
                # our inference backend only support 4D output
                seg_pred = seg_pred.unsqueeze(0)
                return seg_pred
            seg_pred = seg_pred.cpu().numpy()
            # unravel batch dim
            seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred
