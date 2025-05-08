use std::collections::HashMap;

use regex::Regex;

pub fn tokenize_text(text: &str) -> Vec<String> {
    let tokenization_regex = match Regex::new(r#"([,.?_!"()']|--|\s)"#) {
        Ok(regex) => regex,
        Err(_) => panic!("Failed to create regex"),
    };

    tokenization_regex
        .split(text)
        .filter(|token| !token.is_empty())
        .map(|token| token.to_string())
        .collect()
}


pub fn sort_tokens(tokens: &mut Vec<String>) {
    tokens.sort();
}

pub fn create_vocabulary(tokens: &[String]) -> HashMap<usize, String> {
    let mut vocabulary = HashMap::new();
    for (index, token) in tokens.iter().enumerate() {
        vocabulary.insert(index, token.clone());
    }

    vocabulary.insert(tokens.len(), "<|endoftext|>".to_string());
    vocabulary.insert(tokens.len() + 1, "<|unk|>".to_string());

    vocabulary
}

pub struct Tokenizer {
    str_to_int: HashMap<usize, String>,
    int_to_str: HashMap<String, usize>,
}

impl Tokenizer {
    pub fn new(vocab: HashMap<usize, String>) -> Self {
        let mut int_to_str = HashMap::new();
        for (index, token) in vocab.iter() {
            int_to_str.insert(token.clone(), *index);
        }

        Tokenizer {
            str_to_int: vocab,
            int_to_str,
        }
    }


    pub fn encode(&self, text: &str) -> Vec<usize> {
        let tokenization_regex = match Regex::new(r#"([,.?_!"()']|--|\s)"#) {
            Ok(regex) => regex,
            Err(_) => panic!("Failed to create regex"),
        };

        let tokens: Vec<String> = tokenization_regex
            .split(text)
            .filter(|token| !token.is_empty())
            .map(|token| token.to_string())
            .collect();


        let mut encoded = Vec::new();
        for token in tokens {
            if let Some(&index) = self.int_to_str.get(&token) {
                encoded.push(index);
            } else if let Some(&unk_index) = self.int_to_str.get("<|unk|>") {
                encoded.push(unk_index);
            }

        }
        encoded
    }

    pub fn decode(&self, indices: &[usize]) -> String {
        let mut decoded = Vec::new();
        for &index in indices {
            if let Some(token) = self.str_to_int.get(&index) {
                decoded.push(token.clone());
            }
        }
        decoded.join(" ")
    }

}
