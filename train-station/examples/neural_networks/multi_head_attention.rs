//! Multi-Head Attention (MHA) - Minimal example using public API only
//!
//! This example implements a small, self-contained multi-head attention module
//! using public `train_station` APIs and reuses the `LinearLayer` from
//! `basic_linear_layer.rs`. It demonstrates shape-safe forward passes plus a
//! tiny optimization step to verify gradients flow.

use train_station::{
    optimizers::{Adam, Optimizer},
    Tensor,
};

// Reuse the LinearLayer example implementation without duplicating it.
// This pulls in the module locally (its `main` stays namespaced and is unused).
#[path = "basic_linear_layer.rs"]
mod basic_linear_layer;
pub use basic_linear_layer::LinearLayer;

/// Minimal multi-head attention implemented with public API
pub struct MultiHeadAttention {
    pub embed_dim: usize,
    pub num_heads: usize,
    head_dim: usize,
    // Learnable projections
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    out_proj: LinearLayer,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize, seed: Option<u64>) -> Self {
        assert!(
            embed_dim.is_multiple_of(num_heads),
            "embed_dim must be divisible by num_heads"
        );
        let head_dim = embed_dim / num_heads;
        let s0 = seed;
        let s1 = s0.map(|s| s + 1);
        let s2 = s0.map(|s| s + 2);
        let s3 = s0.map(|s| s + 3);
        Self {
            embed_dim,
            num_heads,
            head_dim,
            q_proj: LinearLayer::new(embed_dim, embed_dim, s0),
            k_proj: LinearLayer::new(embed_dim, embed_dim, s1),
            v_proj: LinearLayer::new(embed_dim, embed_dim, s2),
            out_proj: LinearLayer::new(embed_dim, embed_dim, s3),
        }
    }

    /// Collect mutable parameter references for optimization
    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());
        params
    }

    /// Forward pass
    ///
    /// - query: [batch, tgt_len, embed_dim]
    /// - key:   [batch, src_len, embed_dim]
    /// - value: [batch, src_len, embed_dim]
    /// - attn_mask: Optional mask broadcastable to [batch, heads, tgt_len, src_len]
    ///   If provided as a boolean mask (true = keep, false = mask), it will be
    ///   applied via masked_fill with -1e9 before softmax. If provided as tensor
    ///   with other shapes, it is added as is (additive mask).
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Tensor {
        let qkv = Self::project_qkv(query, key, value, &self.q_proj, &self.k_proj, &self.v_proj);
        let (q, k, v) = qkv;

        // Split heads: [b, t, e] -> [b, h, t, d]
        let (b, tq, _e) = Self::triple(query);
        let (_b2, tk, _e2) = Self::triple(key);
        let q = Self::split_heads(&q, b, tq, self.num_heads, self.head_dim);
        let k = Self::split_heads(&k, b, tk, self.num_heads, self.head_dim);
        let v = Self::split_heads(&v, b, tk, self.num_heads, self.head_dim);

        // Scaled dot-product attention
        // logits: [b, h, tq, tk]
        let k_t = k.transpose(2, 3);
        let mut logits = q.matmul(&k_t).div_scalar((self.head_dim as f32).sqrt());
        if let Some(mask) = attn_mask {
            let dims = mask.shape().dims().to_vec();
            // If boolean-like mask matching [b,h,tq,tk], apply masked_fill
            if dims.len() == 4 && dims[0] == b && dims[1] == self.num_heads && dims[2] == tq {
                // Interpret mask > 0.5 as keep; we invert to build masked positions
                let cond: Vec<bool> = mask.data().iter().map(|&v| v < 0.5).collect();
                // Apply masked fill on a flattened view, then reshape back
                let flat_logits = logits.view(vec![(b * self.num_heads * tq * tk) as i32]);
                let filled = flat_logits.masked_fill(&cond, f32::NEG_INFINITY);
                logits = filled.view(vec![b as i32, self.num_heads as i32, tq as i32, tk as i32]);
            } else {
                // Fallback: additive mask
                logits = logits.add_tensor(mask);
            }
        }
        let attn = logits.softmax(3);

        // context: [b, h, tq, d]
        let context = attn.matmul(&v);
        let context = context.permute(vec![0, 2, 1, 3]); // [b, tq, h, d]
        let context = context.contiguous().view(vec![
            b as i32,
            tq as i32,
            (self.num_heads * self.head_dim) as i32,
        ]);

        // Output projection (flatten to 2D, project, then restore 3D)
        let flat = context.view(vec![(b * tq) as i32, self.embed_dim as i32]);
        let out2d = self.out_proj.forward(&flat);
        out2d.view(vec![b as i32, tq as i32, self.embed_dim as i32])
    }

    fn project_qkv(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        q_proj: &LinearLayer,
        k_proj: &LinearLayer,
        v_proj: &LinearLayer,
    ) -> (Tensor, Tensor, Tensor) {
        let (bq, tq, eq) = Self::triple(query);
        let (bk, tk, ek) = Self::triple(key);
        let (_bv, tv, ev) = Self::triple(value);
        assert!(eq == ek && ek == ev, "Q,K,V embed dims must match");
        let q2d = query.view(vec![(bq * tq) as i32, eq as i32]);
        let k2d = key.view(vec![(bk * tk) as i32, ek as i32]);
        let v2d = value.view(vec![(_bv * tv) as i32, ev as i32]);
        let q = q_proj
            .forward(&q2d)
            .view(vec![bq as i32, tq as i32, eq as i32]);
        let k = k_proj
            .forward(&k2d)
            .view(vec![bk as i32, tk as i32, ek as i32]);
        let v = v_proj
            .forward(&v2d)
            .view(vec![bk as i32, tv as i32, ev as i32]);
        (q, k, v)
    }

    fn split_heads(x: &Tensor, b: usize, t: usize, h: usize, d: usize) -> Tensor {
        x.view(vec![b as i32, t as i32, h as i32, d as i32])
            .permute(vec![0, 2, 1, 3])
    }

    fn triple(t: &Tensor) -> (usize, usize, usize) {
        let dims = t.shape().dims();
        assert!(dims.len() == 3, "expected 3D tensor [batch, seq, embed]");
        (dims[0], dims[1], dims[2])
    }
}

#[allow(unused)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Multi-Head Attention Example ===");

    let batch = 2usize;
    let src_len = 5usize;
    let tgt_len = 4usize;
    let embed = 16usize;
    let heads = 4usize;

    let query = Tensor::randn(vec![batch, tgt_len, embed], Some(7));
    let key = Tensor::randn(vec![batch, src_len, embed], Some(8));
    let value = Tensor::randn(vec![batch, src_len, embed], Some(9));

    let mut mha = MultiHeadAttention::new(embed, heads, Some(42));

    // Simple causal mask for target self-attention shape [b, h, tq, tk]
    let mut mask = Tensor::zeros(vec![batch, heads, tgt_len, src_len]);
    // Disallow attending to future positions when tgt_len <= src_len by adding -1e9
    // Here, just demonstrate mask broadcast/add mechanics with a light mask on last head
    if src_len >= tgt_len {
        // set upper triangle to a large negative value for head 0
        for i in 0..tgt_len {
            for j in (i + 1)..src_len {
                let idx = [0usize, 0usize, i, j];
                // Quick set via data_mut using a slice view
                let offset = mask.memory_offset(&idx);
                let data = mask.data_mut();
                data[offset] = -1e9;
            }
        }
    }

    let out = mha.forward(&query, &key, &value, Some(&mask));
    println!("Output shape: {:?}", out.shape().dims());

    // Tiny training step to confirm gradients are wired
    let mut optimizer = Adam::with_learning_rate(0.01);
    let mut params = mha.parameters();
    for p in &params {
        optimizer.add_parameter(p);
    }

    // Dummy loss = mean of output
    let mut loss = out.mean();
    loss.backward(None);
    optimizer.step(&mut params);
    optimizer.zero_grad(&mut params);

    println!("Loss: {:.6}", loss.value());
    println!("=== Done ===");
    Ok(())
}
