use ai_dataloader::Len;
use tch::{nn::{self, Module}, Device, Tensor};
use tiktoken_rs::{get_bpe_from_tokenizer, tokenizer::get_tokenizer};

use crate::{config::Config, data_modifier::GPTDatasetV1, model::Model};

mod architecture;
mod atention;
mod config;
mod data_modifier;
mod file_utils;
mod model;
mod simple_tokenizer;
mod transformer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::new(768, 768, 256, 12, 0.1, 768, 50257, 12, false);

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

    let dataset_train = data_modifier::GPTDatasetV1::new(&tokenizer, train_text, 256, 256);
    let dataset_val  = data_modifier::GPTDatasetV1::new(&tokenizer, val_text, 256, 256);

    let model = Model::new(config);
    let train_size = dataset_train.input_ids.len();

    call_loss_loader(dataset_train, model, train_size);
    Ok(())
}

async fn get_text_from_file(file_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let file_content = file_utils::read_file(file_path).await?;

    Ok(file_content)
}

fn calc_loss_batch(input_batch: Tensor, target_batch: Tensor, model: &Model) -> Tensor {

    let logits = model.forward(&input_batch);

    logits.flatten(0, 1).cross_entropy_for_logits(&target_batch.flatten(0, 0))

}

fn call_loss_loader(dataset: GPTDatasetV1, model: Model, mut num_batch: usize){
    let mut total_loss = 0.;
    let dataset_len = dataset.input_ids.len();
    num_batch = num_batch.min(dataset_len);

    for i in 0..dataset_len {
        let batch_tuple = dataset.get_item(i);

        if i < num_batch {
            let loss = calc_loss_batch(batch_tuple.0, batch_tuple.1, &model);
            loss.print();
        }
    }

}
