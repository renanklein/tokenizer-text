use tokio::{fs::File, io::{AsyncReadExt, AsyncWriteExt}};

pub async fn save_to_file(file_path: &str, content: &str) -> std::io::Result<()> {
    let file = File::create(file_path).await?;

    let mut writer = tokio::io::BufWriter::new(file);
    writer.write_all(content.as_bytes()).await?;

    Ok(())
}



pub async fn read_file(file_path: &str) -> std::io::Result<String> {
    let file = File::open(file_path).await?;
    let mut reader = tokio::io::BufReader::new(file);
    let mut content = String::new();
    reader.read_to_string(&mut content).await?;
    Ok(content)
}

