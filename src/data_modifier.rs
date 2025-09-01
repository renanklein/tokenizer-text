use tch::Tensor;
use tiktoken_rs::CoreBPE;

pub struct GPTDatasetV1 {
    pub input_ids: Vec<Tensor>,
    pub target_ids: Vec<Tensor>,
}

impl GPTDatasetV1 {
     pub fn new(tokenizer: &CoreBPE, text: String, max_length: u32, stride: u32) -> Self {
        let device = tch::Device::cuda_if_available();
        let mut input_ids = Vec::new();
        let mut target_ids = Vec::new();

        let encoded = tokenizer.encode_with_special_tokens(text.as_str());

        for i in (0..(encoded.len() as u32 - max_length)).step_by(stride as usize) {
            // Convert token ids to i64 (Int64) because embeddings and loss functions expect long tensors
            let input_chunk = encoded[i as usize..(i + max_length) as usize]
                .into_iter()
                .map(|x| *x as i64)
                .collect::<Vec<_>>();

            let target_chunk = encoded[(i + 1) as usize..(i + max_length + 1) as usize]
                .into_iter()
                .map(|x| *x as i64)
                .collect::<Vec<_>>();

            let input_tersor = Tensor::from_slice(&input_chunk).to_device(device);
            let target_tensor = Tensor::from_slice(&target_chunk).to_device(device);

            input_ids.push(input_tersor);
            target_ids.push(target_tensor);

        }
         Self {
            input_ids,
            target_ids,
        }
    }

    pub fn get_item(&self, index: usize) -> (Tensor, Tensor) {
        let input = self.input_ids[index].copy().unsqueeze(0);
        let target = self.target_ids[index].copy().unsqueeze(0);
        (input, target)
    }

    pub fn get_input_tensor_batch(&self, batch_size: usize) -> Tensor {
        let input_tensor = Tensor::stack(&self.input_ids[..batch_size], 0);
        input_tensor
    }
}
