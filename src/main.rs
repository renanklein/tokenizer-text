use tiktoken_rs::{get_bpe_from_tokenizer, tokenizer::{self, get_tokenizer}};

mod file_utils;
mod simple_tokenizer;
mod data_modifier;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    let text = get_text_from_file("the-verdict.txt").await?;
    let tokenizer_base = match get_tokenizer("gpt2") {
        Some(tokenizer) => tokenizer,
        None => panic!("Tokenizer not found"),
            
    };

    let tokenizer = match get_bpe_from_tokenizer(tokenizer_base) {
        Ok(tokenizer) => tokenizer,
        Err(e) => panic!("Error getting BPE tokenizer: {}", e),
    };


    let dataset = data_modifier::GPTDatasetV1::new(tokenizer, text, 256, 128);

    let length = dataset.input_ids.len();

    for i in 0..length {
        let input_ids = &dataset.input_ids[i];
        let target_ids = &dataset.target_ids[i];

        println!("Input IDs: {:?}", input_ids);
        println!("Target IDs: {:?}", target_ids);
    }
    
    Ok(())
}


async fn get_text_from_file(file_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let file_content = file_utils::read_file(file_path).await?;
    

    Ok(file_content)
}
