use tch::{nn::{self, VarStore}, Device, Tensor};

pub struct Atention {
    pub query: Tensor,
    pub key: Tensor,
    pub value: Tensor
}

impl Atention {
    pub fn new(d_in: i64, d_out: i64) -> Self {
        let vs = VarStore::new(Device::cuda_if_available());

        let query = vs.root().var("weight", &[d_in, d_out], nn::Init::Randn { mean: 0.0, stdev: 1.0 });
        let key = vs.root().var("weight", &[d_in, d_out], nn::Init::Randn { mean: 0.0, stdev: 1.0 });
        let value = vs.root().var("weight", &[d_in, d_out], nn::Init::Randn { mean: 0.0, stdev: 1.0 });

        Atention {
            query,
            key,
            value
        }
    }


    pub fn forward(&self, input: &Tensor) -> Tensor {
        let queries = input.matmul(&self.query);
        let mut keys = input.matmul(&self.key);
        let values = input.matmul(&self.value);

        let attn_scores = queries.matmul(&keys.t_());
        let scale = (keys.size1().unwrap() as f64).sqrt();
        let scaled_attn_scores = attn_scores / scale;
        let attn_weights = scaled_attn_scores.softmax(-1, tch::Kind::Float);

        let context_vec = attn_weights.matmul(&values);

        context_vec
    }
}
