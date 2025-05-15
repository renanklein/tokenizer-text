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

    let encoded = tokenizer.encode_with_special_tokens(text.as_str());

    let enc_sample = encoded.into_iter().skip(50).collect::<Vec<_>>();

    let context_size  = 4;
    let x = enc_sample.iter().take(context_size).collect::<Vec<_>>();
    let y = enc_sample.iter().skip(1).take(context_size).collect::<Vec<_>>();
    
    println!("x: {:?}", x);
    println!("y:      {:?}", y);


    for i in 1..context_size  {
        let desired = enc_sample[i];
            let current = enc_sample.clone().iter().take(i).map(|x| {
            match  tokenizer.decode(vec![*x]) {
                Ok(decoded) => decoded,
                Err(e) => panic!("Error decoding token: {}", e),
            }
        }).collect::<Vec<_>>().join(" ");

        let desired_decoded = tokenizer.decode(vec![desired]).unwrap_or_else(|_| "Error decoding token".to_string());
        println!("{} -------> {}", current, desired_decoded);
    }

    Ok(())
}


async fn get_text_from_file(file_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let file_content = file_utils::read_file(file_path).await?;
    

    Ok(file_content)
}
