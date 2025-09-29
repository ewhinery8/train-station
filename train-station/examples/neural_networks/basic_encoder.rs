//! Basic Transformer Encoder block using public API and example LinearLayer

use train_station::{
    optimizers::{Adam, Optimizer},
    Tensor,
};

// Reuse LinearLayer from the local example file
#[allow(clippy::duplicate_mod)]
#[path = "basic_linear_layer.rs"]
mod basic_linear_layer;
use basic_linear_layer::LinearLayer;

// Reuse the MHA example module (no duplication)
#[allow(clippy::duplicate_mod)]
#[path = "multi_head_attention.rs"]
mod multi_head_attention;
use multi_head_attention::MultiHeadAttention;

pub struct EncoderBlock {
    pub _embed_dim: usize,
    pub _num_heads: usize,
    mha: MultiHeadAttention,
    ffn_in: LinearLayer,
    ffn_out: LinearLayer,
}

impl EncoderBlock {
    pub fn new(embed_dim: usize, num_heads: usize, seed: Option<u64>) -> Self {
        let s0 = seed;
        let s1 = s0.map(|s| s + 1);
        let s2 = s0.map(|s| s + 2);
        Self {
            _embed_dim: embed_dim,
            _num_heads: num_heads,
            mha: MultiHeadAttention::new(embed_dim, num_heads, s0),
            ffn_in: LinearLayer::new(embed_dim, embed_dim * 2, s1),
            ffn_out: LinearLayer::new(embed_dim * 2, embed_dim, s2),
        }
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.mha.parameters());
        params.extend(self.ffn_in.parameters());
        params.extend(self.ffn_out.parameters());
        params
    }

    /// Forward pass
    /// input: [batch, seq, embed]
    /// attn_mask: optional mask broadcastable to [batch, heads, seq, seq]
    pub fn forward(&self, input: &Tensor, attn_mask: Option<&Tensor>) -> Tensor {
        let attn = self.mha.forward(input, input, input, attn_mask);
        let res1 = attn.add_tensor(input);

        // Feed-forward network with ReLU and residual
        let (b, t, e) = Self::triple(input);
        let x2d = res1.contiguous().view(vec![(b * t) as i32, e as i32]);
        let hidden = self.ffn_in.forward(&x2d).relu();
        let out2d = self.ffn_out.forward(&hidden);
        let out = out2d.view(vec![b as i32, t as i32, e as i32]);
        out.add_tensor(&res1)
    }

    fn triple(t: &Tensor) -> (usize, usize, usize) {
        let d = t.shape().dims();
        (d[0], d[1], d[2])
    }
}

#[allow(unused)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Basic Encoder Example ===");

    let batch = 2usize;
    let seq = 6usize;
    let embed = 32usize;
    let heads = 4usize;

    let input = Tensor::randn(vec![batch, seq, embed], Some(11));
    let mut enc = EncoderBlock::new(embed, heads, Some(123));

    // Example: no mask (set Some(mask) to use masking)
    let out = enc.forward(&input, None);
    println!("Output shape: {:?}", out.shape().dims());

    // Verify gradients/optimization
    let mut opt = Adam::with_learning_rate(0.01);
    let mut params = enc.parameters();
    for p in &params {
        opt.add_parameter(p);
    }
    let mut loss = out.mean();
    loss.backward(None);
    opt.step(&mut params);
    opt.zero_grad(&mut params);
    println!("Loss: {:.6}", loss.value());
    println!("=== Done ===");
    Ok(())
}
