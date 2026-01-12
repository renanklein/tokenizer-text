use tch::{nn::{AdamW, Module, OptimizerConfig, VarStore}, Device, Tensor};
use std::{env, fs::read_to_string};
use rand::{rng, seq::SliceRandom};

use crate::{config::Config, model::Model, data_modifier::GPTDatasetV1};

mod architecture;
mod atention;
mod config;
mod data_modifier;
mod file_utils;
mod model;
mod transformer;


const LEARNING_RATE: f64 = 0.0003;
const EPOCHS: i64 = 10;
const BLOCK_SIZE: usize = 32; // Context window size
const BATCH_SIZE: usize = 4;
const STRIDE: usize = 4;


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::new(768, 768, 256, 12, 0.1, 768, 50257, 12, false);

    let text_data = read_to_string("the-verdict.txt")?;

    let device = Device::cuda_if_available();

    let vs = VarStore::new(device);
    let root = vs.root();
    let model = Model::new(&root, config);

    let tokenizer = model.get_tokenizer();
    let dataset = GPTDatasetV1::new(&tokenizer, text_data, BLOCK_SIZE as u32, STRIDE as u32);

    let max_steps: usize = env::var("MAX_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);

    train(&dataset, &model, &vs, device, max_steps);

    let prompt = String::from("Every effort moves you");

    let result_logits = model.generate_text(model.text_to_tensor(prompt), 15, 256, 1.4, 25);

    let result = model.logits_to_text(result_logits);

    println!("{}", result);


    Ok(())
}

fn calc_loss_batch(input_batch: &Tensor, target_batch: &Tensor, model: &Model) -> Tensor {

    let logits = model.forward(input_batch);

    let targets = target_batch.to_kind(tch::Kind::Int64).flatten(0, 1);


    logits.flatten(0, 1).cross_entropy_for_logits(&targets)

}


fn train (dataset: &GPTDatasetV1, model: &Model, vs: &VarStore, device: Device, max_steps: usize) {
    let mut optimizer = AdamW::default().build(vs, LEARNING_RATE).unwrap();

    let mut idx = 0;

    for epoch in 1 .. (1 + EPOCHS) {

        let mut sum_loss = 0.;
        let mut cnt_loss = 0.;
        
        // Create a shuffled list of indices
        let mut indices: Vec<usize> = (0..dataset.input_ids.len()).collect();
        indices.shuffle(&mut rng());

        // Iterate in batches
        for chunk in indices.chunks(BATCH_SIZE) {
            let mut input_tensors = Vec::new();
            let mut target_tensors = Vec::new();

            for &i in chunk {
                input_tensors.push(dataset.input_ids[i].shallow_clone());
                target_tensors.push(dataset.target_ids[i].shallow_clone());
            }

            // Stack tensors to create a batch: [BATCH_SIZE, BLOCK_SIZE]
            let input_batch = Tensor::stack(&input_tensors, 0).to_device(device);
            let target_batch = Tensor::stack(&target_tensors, 0).to_device(device);

            let loss = calc_loss_batch(&input_batch, &target_batch, &model);

            loss.print();
            optimizer.backward_step_clip(&loss, 0.5);

            sum_loss += f64::try_from(loss).unwrap();
            cnt_loss += 1.0;
            idx += 1;

            if idx % 10 == 0 { // Print more often for small dataset
                println!("Epoch : {:?} step {} loss {:5.3}", epoch, idx, sum_loss/cnt_loss);
            }

            if idx >= max_steps {
                println!("Reached max_steps {}. Stopping early to avoid OOM.", max_steps);
                return;
            }
        }
    } 
}
