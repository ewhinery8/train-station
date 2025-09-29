//! Basic Transformer Decoder block using public API and example modules

use train_station::{
    optimizers::{Adam, Optimizer},
    Tensor,
};

#[path = "basic_linear_layer.rs"]
mod basic_linear_layer;
use basic_linear_layer::LinearLayer;

#[allow(clippy::duplicate_mod)]
#[path = "multi_head_attention.rs"]
mod multi_head_attention;
use multi_head_attention::MultiHeadAttention;

pub struct DecoderBlock {
    pub _embed_dim: usize,
    pub _num_heads: usize,
    self_attn: MultiHeadAttention,
    cross_attn: MultiHeadAttention,
    ffn_in: LinearLayer,
    ffn_out: LinearLayer,
}

impl DecoderBlock {
    pub fn new(embed_dim: usize, num_heads: usize, seed: Option<u64>) -> Self {
        let s0 = seed;
        let s1 = s0.map(|s| s + 1);
        let s2 = s0.map(|s| s + 2);
        let s3 = s0.map(|s| s + 3);
        Self {
            _embed_dim: embed_dim,
            _num_heads: num_heads,
            self_attn: MultiHeadAttention::new(embed_dim, num_heads, s0),
            cross_attn: MultiHeadAttention::new(embed_dim, num_heads, s1),
            ffn_in: LinearLayer::new(embed_dim, embed_dim * 2, s2),
            ffn_out: LinearLayer::new(embed_dim * 2, embed_dim, s3),
        }
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.cross_attn.parameters());
        params.extend(self.ffn_in.parameters());
        params.extend(self.ffn_out.parameters());
        params
    }

    /// Forward pass
    /// tgt: [batch, tgt_len, embed]
    /// memory: [batch, src_len, embed] (encoder outputs)
    /// causal_mask: mask broadcastable to [batch, heads, tgt_len, tgt_len] (true=keep, false=masked)
    /// cross_mask:  optional mask broadcastable to [batch, heads, tgt_len, src_len]
    pub fn forward(
        &self,
        tgt: &Tensor,
        memory: &Tensor,
        causal_mask: Option<&Tensor>,
        cross_mask: Option<&Tensor>,
    ) -> Tensor {
        let self_attn = self.self_attn.forward(tgt, tgt, tgt, causal_mask);
        let res1 = self_attn.add_tensor(tgt);

        let cross = self.cross_attn.forward(&res1, memory, memory, cross_mask);
        let res2 = cross.add_tensor(&res1);

        let (b, t, e) = Self::triple(tgt);
        let x2d = res2.contiguous().view(vec![(b * t) as i32, e as i32]);
        let hidden = self.ffn_in.forward(&x2d).relu();
        let out2d = self.ffn_out.forward(&hidden);
        let out = out2d.view(vec![b as i32, t as i32, e as i32]);
        out.add_tensor(&res2)
    }

    fn triple(t: &Tensor) -> (usize, usize, usize) {
        let d = t.shape().dims();
        (d[0], d[1], d[2])
    }
}

#[allow(unused)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Basic Decoder Example ===");

    let batch = 2usize;
    let src = 7usize;
    let tgt = 5usize;
    let embed = 32usize;
    let heads = 4usize;

    let memory = Tensor::randn(vec![batch, src, embed], Some(21));
    let tgt_in = Tensor::randn(vec![batch, tgt, embed], Some(22));

    let mut dec = DecoderBlock::new(embed, heads, Some(456));
    let out = dec.forward(&tgt_in, &memory, None, None);
    println!("Output shape: {:?}", out.shape().dims());

    let mut opt = Adam::with_learning_rate(0.01);
    let mut params = dec.parameters();
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
