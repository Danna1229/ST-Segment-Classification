import torch
import torch.nn as nn

# Define 1-D convolution block
class unet_1d_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(unet_1d_conv_block, self).__init__()
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, data_input):
        data_input = self.conv1d_block(data_input)
        return data_input


# Define continuous convolution blocks
class unet_1d_conv_group(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, padding):
        super(unet_1d_conv_group, self).__init__()
        self.conv1d_block1 = unet_1d_conv_block(in_channels, mid_channels, kernel_size, stride, padding)
        self.conv1d_block2 = unet_1d_conv_block(mid_channels, out_channels, kernel_size, stride, padding)

    def forward(self, data_input):
        middle = self.conv1d_block1(data_input)
        return self.conv1d_block2(middle)


# Define the complete unet_1d model
class unet_1d_model(nn.Module):
    def __init__(self):
        super(unet_1d_model, self).__init__()

        self.unet_encoder1 = unet_1d_conv_group(1, 16, 16, 9, 1, 4)
        self.unet_encoder2 = unet_1d_conv_group(16, 32, 32, 9, 1, 4)
        self.unet_encoder3 = unet_1d_conv_group(32, 64, 64, 9, 1, 4)
        self.unet_encoder4 = unet_1d_conv_group(64, 128, 128, 9, 1, 4)
        self.unet_encoder5 = unet_1d_conv_group(128, 256, 256, 9, 1, 4)

        self.max_pool1d_1 = nn.MaxPool1d(8, 2, 3)
        self.max_pool1d_2 = nn.MaxPool1d(8, 2, 3)
        self.max_pool1d_3 = nn.MaxPool1d(8, 2, 3)
        self.max_pool1d_4 = nn.MaxPool1d(8, 2, 3)

        self.tran_conv1 = nn.ConvTranspose1d(256, 256, 8, 2, 3)
        self.tran_conv2 = nn.ConvTranspose1d(128, 128, 8, 2, 3)
        self.tran_conv3 = nn.ConvTranspose1d(64, 64, 8, 2, 3)
        self.tran_conv4 = nn.ConvTranspose1d(32, 32, 8, 2, 3)

        self.unet_decoder1 = unet_1d_conv_group(384, 128, 128, 3, 1, 1)
        self.unet_decoder2 = unet_1d_conv_group(192, 64, 64, 3, 1, 1)
        self.unet_decoder3 = unet_1d_conv_group(96, 32, 32, 3, 1, 1)
        self.unet_decoder4 = unet_1d_conv_group(48, 4, 1, 3, 1, 1)

        self.classifier_1 = nn.Sequential(
            nn.Linear(230400, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 3),
            nn.Softmax(dim=1),
        )

        self.classifier_12 = nn.Sequential(
            nn.Linear(230403, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 12),
            nn.Sigmoid(),
        )

    def forward(self, data_input):
        encoders = []
        decoders = []
        for i in range(12):
            input_i = data_input[:, i, :].unsqueeze(1)

            encoder_1_out = self.unet_encoder1(input_i)
            pool_1_out = self.max_pool1d_1(encoder_1_out)

            encoder_2_out = self.unet_encoder2(pool_1_out)
            pool_2_out = self.max_pool1d_2(encoder_2_out)

            encoder_3_out = self.unet_encoder3(pool_2_out)
            pool_3_out = self.max_pool1d_3(encoder_3_out)

            encoder_4_out = self.unet_encoder4(pool_3_out)
            pool_4_out = self.max_pool1d_4(encoder_4_out)

            encoder_5_out = self.unet_encoder5(pool_4_out)

            up_conv_1_out = self.tran_conv1(encoder_5_out)
            decoder1_in = torch.cat((encoder_4_out, up_conv_1_out), dim=1)
            decoder1_out = self.unet_decoder1(decoder1_in)

            up_conv_2_out = self.tran_conv2(decoder1_out)
            decoder2_in = torch.cat((encoder_3_out, up_conv_2_out), dim=1)
            decoder2_out = self.unet_decoder2(decoder2_in)

            up_conv_3_out = self.tran_conv3(decoder2_out)
            decoder3_in = torch.cat((encoder_2_out, up_conv_3_out), dim=1)
            decoder3_out = self.unet_decoder3(decoder3_in)

            up_conv_4_out = self.tran_conv4(decoder3_out)
            decoder4_in = torch.cat((encoder_1_out, up_conv_4_out), dim=1)
            decoder4_out = self.unet_decoder4(decoder4_in)

            encoders.append(encoder_5_out)
            decoders.append(decoder4_out)

        encoders_cat = torch.cat(encoders, dim=1)
        decoders_cat = torch.cat(decoders, dim=1)


        fc_input=encoders_cat.view(encoders_cat.size(0), -1)

        multi_class_predictions = self.classifier_1(fc_input)

        # The prediction results of the multi-class classification are passed to the multi-label classifier
        multi_label_input = torch.cat([fc_input, multi_class_predictions], dim=1)
        multi_label_predictions = self.classifier_12(multi_label_input)


        return encoders_cat, decoders_cat, multi_class_predictions, multi_label_predictions

    def cal_loss_Unet(self, decoding, signals):
        Unet_loss = nn.MSELoss()(decoding, signals)
        return Unet_loss

    def cal_loss_multi_class(self, pred_label, tar_label):
        multi_class_loss = nn.CrossEntropyLoss()(pred_label, tar_label.squeeze(dim=1))
        return multi_class_loss

    def cal_loss_multi_label(self, pred_label, tar_label):
        multi_label_loss = nn.BCEWithLogitsLoss()(pred_label, tar_label.float())
        return multi_label_loss

if __name__ == '__main__':
    # Model testing
    test_input = torch.randn(32, 12, 1200)
    test_unet_model = unet_1d_model()
    encoders_cat, decoders_cat, multi_class_predictions, multi_label_predictions = test_unet_model.forward(test_input)
    print(decoders_cat.shape)