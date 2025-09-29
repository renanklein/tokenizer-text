use tch::{data::TextData, nn::{self, AdamW, Module, OptimizerConfig, VarStore}, Device, Kind, Tensor};
use tiktoken_rs::{get_bpe_from_tokenizer, tokenizer::get_tokenizer};
use std::env;

use crate::{config::Config, data_modifier::GPTDatasetV1, model::Model};

mod architecture;
mod atention;
mod config;
mod data_modifier;
mod file_utils;
mod model;
mod simple_tokenizer;
mod transformer;


const LEARNING_RATE: f64 = 0.0003;
const EPOCHS: i64 = 10;
const BLOCK_SIZE: i64 = 128; // Context window size for training chunks


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let small_model = env::var("SMALL_MODEL").is_ok();
    let config = if small_model {
        Config::new(256, 256, 128, 4, 0.1, 256, 50257, 4, false)
    } else {
        Config::new(768, 768, 256, 12, 0.1, 768, 50257, 12, false)
    };

    let text = get_text_from_file("the-verdict.txt").await?;

    let train_ratio = 0.9;

    let total_chars = text.chars().count();
    let split_idx = (total_chars as f64 * train_ratio).round() as usize;
    let train_text: String = text.chars().take(split_idx).collect();
    let val_text: String = text.chars().skip(split_idx).collect();


    let tokenizer_base = match get_tokenizer("gpt2") {
        Some(tokenizer) => tokenizer,
        None => panic!("Tokenizer not found"),
    };

    let tokenizer = match get_bpe_from_tokenizer(tokenizer_base) {
        Ok(tokenizer) => tokenizer,
        Err(e) => panic!("Error getting BPE tokenizer: {}", e),
    };


    let data = TextData::new("the-verdict.txt")?;

    let device = Device::cuda_if_available();

    let vs = VarStore::new(device);
    let root = vs.root();
    let model = Model::new(&root, config);

    let max_steps: usize = env::var("MAX_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500); // default

    train(data, &model, &vs, device, max_steps);



    let dataset_train = data_modifier::GPTDatasetV1::new(&tokenizer, train_text, 256, 256);
    let dataset_val  = data_modifier::GPTDatasetV1::new(&tokenizer, val_text, 256, 256);

    let train_size = dataset_train.input_ids.len();



    Ok(())
}

async fn get_text_from_file(file_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let file_content = file_utils::read_file(file_path).await?;

    Ok(file_content)
}

fn calc_loss_batch(input_batch: &Tensor, target_batch: &Tensor, model: &Model) -> Tensor {

    let logits = model.forward(input_batch);

    let targets = target_batch.to_kind(tch::Kind::Int64).flatten(0, 1);


    logits.flatten(0, 1).cross_entropy_for_logits(&targets)

}


fn train (data: TextData, model: &Model, vs: &VarStore, device: Device, max_steps: usize) {
    let mut optimizer = AdamW::default().build(vs, LEARNING_RATE).unwrap();
    // Removed weight decay group settings to avoid out-of-range panic; customize groups if added later.


    let mut idx = 0;

    for epoch in 1 .. (1 + EPOCHS) {

        let mut sum_loss = 0.;
        let mut cnt_loss = 0.;

        for batch in data.iter_shuffle(BLOCK_SIZE + 1, BLOCK_SIZE) {

            let input_batch = batch.narrow(1, 0, BLOCK_SIZE).to_kind(Kind::Int64).to_device(device);
            let target_batch = batch.narrow(1, 1, BLOCK_SIZE).to_kind(Kind::Int64).to_device(device);

            let loss = calc_loss_batch(&input_batch, &target_batch, &model);

            loss.print();
            optimizer.backward_step_clip(&loss, 0.5);

            sum_loss += f64::try_from(loss).unwrap();
            cnt_loss += 1.0;
            idx += 1;

            if idx % 10000 == 0 {
                println!("Epoch : {:?} step {} loss {:5.3}", epoch, idx, sum_loss/cnt_loss);
            }

            if idx >= max_steps {
                println!("Reached max_steps {}. Stopping early to avoid OOM.", max_steps);
                return;
            }


        }


    } 


}
