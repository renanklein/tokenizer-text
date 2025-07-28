pub struct Config {
    pub d_in: i64,
    pub d_out: i64,
    pub context_length: i64,
    pub num_heads: i64,
    pub drop_rate: f64,
    pub emb_dim: i64,
    pub qkv_bias: bool,
}

impl Config {
    pub fn new(
        d_in: i64,
        d_out: i64,
        context_length: i64,
        num_heads: i64,
        drop_rate: f64,
        emb_dim: i64,
        qkv_bias: bool,
    ) -> Self {
        Self {
            d_in,
            d_out,
            context_length,
            num_heads,
            drop_rate,
            emb_dim,
            qkv_bias,
        }
    }
}
