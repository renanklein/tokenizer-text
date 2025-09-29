use std::f64::consts::PI;

use tch::{nn::{self, linear, seq, Module, Sequential}, Kind, Tensor};

#[derive(Debug)]
pub struct LayerNorm {
    eps: f64,
    shift: Tensor,
    scale: Tensor,
}


#[derive(Debug)]
pub struct GELU;

#[derive(Debug)]
pub struct FeedForward {
    emb_dim: i64,
    layers: Sequential
}

impl LayerNorm {
    pub fn new(path: &nn::Path, emb_dim: i64) -> Self {
        let eps = 1e-5;
        let scale = path.ones("scale", &[emb_dim]);
        let shift = path.zeros("shift", &[emb_dim]);
        LayerNorm { eps, shift, scale }
    }

}

impl nn::Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mean = xs.mean_dim(Some(&[-1i64][..]), true, Kind::Float);
        let var = xs.var_dim(Some(&[-1i64][..]), false, true);

        let norm_x = (xs - mean)/Tensor::sqrt(&(self.eps + var));

        &self.scale * norm_x + &self.shift
    }

}

impl GELU {
    pub fn new() -> Self {
        GELU
    }

}

impl nn::Module for GELU {

    fn forward(&self, xs: &Tensor) -> Tensor{
        let part = Tensor::from(2.0 / PI);

        0.5 * xs * (1.0 + Tensor::tanh(&(&part * &(xs + 0.044715 + xs.pow_tensor_scalar(3.0)))))

    }
}

impl FeedForward {
    pub fn new(path: &nn::Path, emb_dim: i64) -> Self {
        let layer1 = linear(path, emb_dim, 4 * emb_dim, Default::default());
        let layer2 = linear(path, 4 * emb_dim, emb_dim, Default::default());
        let layers = seq().add(layer1).add(GELU::new()).add(layer2);
        FeedForward { emb_dim, layers }
    }

}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.layers.forward(xs)
    }
}
