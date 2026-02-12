// Generated from ONNX "segmentation_3_1.onnx" by burn-import
use burn::prelude::*;
use burn::nn::BiLstm;
use burn::nn::BiLstmConfig;
use burn::nn::InstanceNorm;
use burn::nn::InstanceNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::LstmState;
use burn::nn::PaddingConfig1d;
use burn::nn::conv::Conv1d;
use burn::nn::conv::Conv1dConfig;
use burn::nn::pool::MaxPool1d;
use burn::nn::pool::MaxPool1dConfig;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    constant33: burn::module::Param<Tensor<B, 3>>,
    instancenormalization1: InstanceNorm<B>,
    conv1d1: Conv1d<B>,
    maxpool1d1: MaxPool1d,
    instancenormalization2: InstanceNorm<B>,
    conv1d2: Conv1d<B>,
    maxpool1d2: MaxPool1d,
    instancenormalization3: InstanceNorm<B>,
    conv1d3: Conv1d<B>,
    maxpool1d3: MaxPool1d,
    instancenormalization4: InstanceNorm<B>,
    lstm1: BiLstm<B>,
    lstm2: BiLstm<B>,
    lstm3: BiLstm<B>,
    lstm4: BiLstm<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    device: burn::module::Ignored<B::Device>,
}


impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file("model/model.bpk", &Default::default())
    }
}

impl<B: Backend> Model<B> {
    /// Load model weights from a burnpack file.
    pub fn from_file(file: &str, device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(file);
        model.load_from(&mut store).expect("Failed to load burnpack file");
        model
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant33: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([2, 1, 128], device),
            device.clone(),
            false,
            [2, 1, 128].into(),
        );
        let instancenormalization1 = InstanceNormConfig::new(1)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let conv1d1 = Conv1dConfig::new(1, 80, 251)
            .with_stride(10)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let maxpool1d1 = MaxPool1dConfig::new(3)
            .with_stride(3)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_ceil_mode(false)
            .init();
        let instancenormalization2 = InstanceNormConfig::new(80)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let conv1d2 = Conv1dConfig::new(80, 60, 5)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool1d2 = MaxPool1dConfig::new(3)
            .with_stride(3)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_ceil_mode(false)
            .init();
        let instancenormalization3 = InstanceNormConfig::new(60)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let conv1d3 = Conv1dConfig::new(60, 60, 5)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool1d3 = MaxPool1dConfig::new(3)
            .with_stride(3)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_ceil_mode(false)
            .init();
        let instancenormalization4 = InstanceNormConfig::new(60)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let lstm1 = BiLstmConfig::new(60, 128, true)
            .with_batch_first(false)
            .with_input_forget(false)
            .init(device);
        let lstm2 = BiLstmConfig::new(256, 128, true)
            .with_batch_first(false)
            .with_input_forget(false)
            .init(device);
        let lstm3 = BiLstmConfig::new(256, 128, true)
            .with_batch_first(false)
            .with_input_forget(false)
            .init(device);
        let lstm4 = BiLstmConfig::new(256, 128, true)
            .with_batch_first(false)
            .with_input_forget(false)
            .init(device);
        let linear1 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let linear2 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear3 = LinearConfig::new(128, 3).with_bias(true).init(device);
        Self {
            constant33,
            instancenormalization1,
            conv1d1,
            maxpool1d1,
            instancenormalization2,
            conv1d2,
            maxpool1d2,
            instancenormalization3,
            conv1d3,
            maxpool1d3,
            instancenormalization4,
            lstm1,
            lstm2,
            lstm3,
            lstm4,
            linear1,
            linear2,
            linear3,
            phantom: core::marker::PhantomData,
            device: burn::module::Ignored(device.clone()),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input_audio: Tensor<B, 3>) -> Tensor<B, 3> {
        let constant33_out1 = self.constant33.val();
        let instancenormalization1_out1 = self
            .instancenormalization1
            .forward(input_audio);
        let conv1d1_out1 = self.conv1d1.forward(instancenormalization1_out1);
        let abs1_out1 = conv1d1_out1.abs();
        let maxpool1d1_out1 = self.maxpool1d1.forward(abs1_out1);
        let instancenormalization2_out1 = self
            .instancenormalization2
            .forward(maxpool1d1_out1);
        let leakyrelu1_out1 = burn::tensor::activation::leaky_relu(
            instancenormalization2_out1,
            0.009999999776482582,
        );
        let conv1d2_out1 = self.conv1d2.forward(leakyrelu1_out1);
        let maxpool1d2_out1 = self.maxpool1d2.forward(conv1d2_out1);
        let instancenormalization3_out1 = self
            .instancenormalization3
            .forward(maxpool1d2_out1);
        let leakyrelu2_out1 = burn::tensor::activation::leaky_relu(
            instancenormalization3_out1,
            0.009999999776482582,
        );
        let conv1d3_out1 = self.conv1d3.forward(leakyrelu2_out1);
        let maxpool1d3_out1 = self.maxpool1d3.forward(conv1d3_out1);
        let instancenormalization4_out1 = self
            .instancenormalization4
            .forward(maxpool1d3_out1);
        let leakyrelu3_out1 = burn::tensor::activation::leaky_relu(
            instancenormalization4_out1,
            0.009999999776482582,
        );
        let transpose1_out1 = leakyrelu3_out1.permute([2, 0, 1]);
        let (lstm1_out1, _lstm1_out2, _lstm1_out3) = {
            let (output_seq, final_state) = self
                .lstm1
                .forward(
                    transpose1_out1,
                    Some(
                        LstmState::new(constant33_out1.clone(), constant33_out1.clone()),
                    ),
                );
            (
                {
                    let [seq_len, batch_size, _] = output_seq.dims();
                    let reshaped = output_seq
                        .reshape([seq_len, batch_size, 2, 128usize]);
                    reshaped.swap_dims(1, 2)
                },
                final_state.hidden,
                final_state.cell,
            )
        };
        let transpose2_out1 = lstm1_out1.permute([0, 2, 1, 3]);
        let reshape1_out1 = transpose2_out1.reshape([0, 0, -1]);
        let (lstm2_out1, _lstm2_out2, _lstm2_out3) = {
            let (output_seq, final_state) = self
                .lstm2
                .forward(
                    reshape1_out1,
                    Some(
                        LstmState::new(constant33_out1.clone(), constant33_out1.clone()),
                    ),
                );
            (
                {
                    let [seq_len, batch_size, _] = output_seq.dims();
                    let reshaped = output_seq
                        .reshape([seq_len, batch_size, 2, 128usize]);
                    reshaped.swap_dims(1, 2)
                },
                final_state.hidden,
                final_state.cell,
            )
        };
        let transpose3_out1 = lstm2_out1.permute([0, 2, 1, 3]);
        let reshape2_out1 = transpose3_out1.reshape([0, 0, -1]);
        let (lstm3_out1, _lstm3_out2, _lstm3_out3) = {
            let (output_seq, final_state) = self
                .lstm3
                .forward(
                    reshape2_out1,
                    Some(
                        LstmState::new(constant33_out1.clone(), constant33_out1.clone()),
                    ),
                );
            (
                {
                    let [seq_len, batch_size, _] = output_seq.dims();
                    let reshaped = output_seq
                        .reshape([seq_len, batch_size, 2, 128usize]);
                    reshaped.swap_dims(1, 2)
                },
                final_state.hidden,
                final_state.cell,
            )
        };
        let transpose4_out1 = lstm3_out1.permute([0, 2, 1, 3]);
        let reshape3_out1 = transpose4_out1.reshape([0, 0, -1]);
        let (lstm4_out1, _lstm4_out2, _lstm4_out3) = {
            let (output_seq, final_state) = self
                .lstm4
                .forward(
                    reshape3_out1,
                    Some(LstmState::new(constant33_out1.clone(), constant33_out1.clone())),
                );
            (
                {
                    let [seq_len, batch_size, _] = output_seq.dims();
                    let reshaped = output_seq
                        .reshape([seq_len, batch_size, 2, 128usize]);
                    reshaped.swap_dims(1, 2)
                },
                final_state.hidden,
                final_state.cell,
            )
        };
        let transpose5_out1 = lstm4_out1.permute([0, 2, 1, 3]);
        let reshape4_out1 = transpose5_out1.reshape([0, 0, -1]);
        let transpose6_out1 = reshape4_out1.permute([1, 0, 2]);
        let linear1_out1 = self.linear1.forward(transpose6_out1);
        let leakyrelu4_out1 = burn::tensor::activation::leaky_relu(
            linear1_out1,
            0.009999999776482582,
        );
        let linear2_out1 = self.linear2.forward(leakyrelu4_out1);
        let leakyrelu5_out1 = burn::tensor::activation::leaky_relu(
            linear2_out1,
            0.009999999776482582,
        );
        let linear3_out1 = self.linear3.forward(leakyrelu5_out1);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(linear3_out1);
        sigmoid1_out1
    }
}
