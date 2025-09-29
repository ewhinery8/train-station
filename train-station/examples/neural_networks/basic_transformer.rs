//! Basic Transformer (Encoder-Decoder) wiring using encoder/decoder examples

use train_station::{
    optimizers::{Adam, Optimizer},
    Tensor,
};

#[path = "basic_encoder.rs"]
mod basic_encoder;
use basic_encoder::EncoderBlock;

#[path = "basic_decoder.rs"]
mod basic_decoder;
use basic_decoder::DecoderBlock;

pub struct BasicTransformer {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    encoders: Vec<EncoderBlock>,
    decoders: Vec<DecoderBlock>,
}

impl BasicTransformer {
    pub fn new(embed_dim: usize, num_heads: usize, num_layers: usize, seed: Option<u64>) -> Self {
        let mut encoders = Vec::new();
        let mut decoders = Vec::new();
        for i in 0..num_layers {
            encoders.push(EncoderBlock::new(
                embed_dim,
                num_heads,
                seed.map(|s| s + i as u64),
            ));
            decoders.push(DecoderBlock::new(
                embed_dim,
                num_heads,
                seed.map(|s| s + 100 + i as u64),
            ));
        }
        Self {
            embed_dim,
            num_heads,
            num_layers,
            encoders,
            decoders,
        }
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for e in &mut self.encoders {
            params.extend(e.parameters());
        }
        for d in &mut self.decoders {
            params.extend(d.parameters());
        }
        params
    }

    /// Forward pass
    /// src: [batch, src_len, embed]
    /// tgt: [batch, tgt_len, embed]
    pub fn forward(&self, src: &Tensor, tgt: &Tensor) -> Tensor {
        let mut memory = src.clone();
        for enc in &self.encoders {
            memory = enc.forward(&memory, None);
        }
        let mut out = tgt.clone();
        for dec in &self.decoders {
            out = dec.forward(&out, &memory, None, None);
        }
        out
    }

    /// Greedy auto-regressive inference (toy)
    pub fn infer_autoregressive(&self, src: &Tensor, max_steps: usize) -> Tensor {
        let (b, _s, e) = Self::triple(src);
        let mut memory = src.clone();
        for enc in &self.encoders {
            memory = enc.forward(&memory, None);
        }

        let mut out_seq: Vec<Tensor> = Vec::new();
        // Start token: zeros
        let mut current = Tensor::zeros(vec![b, 1, e]);
        for _step in 0..max_steps {
            // Build causal mask for length t
            let t = current.shape().dims()[1];
            let mut causal = Tensor::ones(vec![b, self.num_heads, t, t]);
            // Upper triangle as false -> masked for all batches and heads
            for bb in 0..b {
                for hh in 0..self.num_heads {
                    for i in 0..t {
                        for j in (i + 1)..t {
                            let offset = causal.memory_offset(&[bb, hh, i, j]);
                            let data = causal.data_mut();
                            data[offset] = 0.0;
                        }
                    }
                }
            }
            let mut step_out = current.clone();
            for dec in &self.decoders {
                step_out = dec.forward(&step_out, &memory, Some(&causal), None);
            }
            // (Toy) append placeholder token; real models would project last token
            out_seq.push(step_out.clone());
            // Append a zero token to grow sequence by 1 for next causal computation
            current = Tensor::zeros(vec![b, t + 1, e]);
        }
        // Simple return of final sequence placeholder
        current
    }

    /// Non auto-regressive inference: single forward pass
    pub fn infer_non_autoregressive(&self, src: &Tensor, tgt_len: usize) -> Tensor {
        let (b, _s, e) = Self::triple(src);
        let mut memory = src.clone();
        for enc in &self.encoders {
            memory = enc.forward(&memory, None);
        }
        let tgt = Tensor::zeros(vec![b, tgt_len, e]);
        let mut out = tgt.clone();
        for dec in &self.decoders {
            out = dec.forward(&out, &memory, None, None);
        }
        out
    }

    /// Helper: build boolean-like causal mask [b, heads, t, t] with 1.0 keep, 0.0 masked
    fn build_causal_mask_static(batch: usize, heads: usize, t: usize) -> Tensor {
        let mut mask = Tensor::ones(vec![batch, heads, t, t]);
        for bb in 0..batch {
            for hh in 0..heads {
                for i in 0..t {
                    for j in (i + 1)..t {
                        let offset = mask.memory_offset(&[bb, hh, i, j]);
                        let data = mask.data_mut();
                        data[offset] = 0.0;
                    }
                }
            }
        }
        mask
    }

    /// Non auto-regressive training with teacher forcing (single pass)
    pub fn train_non_autoregressive_steps(
        &mut self,
        src: &Tensor,
        tgt: &Tensor,
        steps: usize,
        lr: f32,
    ) {
        let mut opt = Adam::with_learning_rate(lr);
        {
            let params_once = self.parameters();
            for p in &params_once {
                opt.add_parameter(p);
            }
        }
        for step in 0..steps {
            // forward + backward scope (immutable borrow)
            {
                let pred = self.forward(src, tgt);
                let diff = pred.sub_tensor(tgt);
                let mut loss = diff.pow_scalar(2.0).mean();
                if step == 0 || step + 1 == steps {
                    println!("NAR train step {}: loss={:.6}", step, loss.value());
                }
                loss.backward(None);
            }
            // step + zero_grad scope (mutable borrow)
            let mut params_step = self.parameters();
            opt.step(&mut params_step);
            opt.zero_grad(&mut params_step);
        }
    }

    /// Auto-regressive training (teacher forcing): predict next token with causal mask
    pub fn train_autoregressive_steps(
        &mut self,
        src: &Tensor,
        tgt: &Tensor,
        steps: usize,
        lr: f32,
    ) {
        let mut opt = Adam::with_learning_rate(lr);
        {
            let params_once = self.parameters();
            for p in &params_once {
                opt.add_parameter(p);
            }
        }

        // Build encoder memory once (static dataset demo)
        let mut memory = src.clone();
        for enc in &self.encoders {
            memory = enc.forward(&memory, None);
        }

        let (b, t, _e) = Self::triple(tgt);
        // Predict y[t] from y[:t] using causal mask; here we simply predict full seq with mask
        let causal = Self::build_causal_mask_static(b, self.num_heads, t);
        for step in 0..steps {
            // forward + backward scope
            {
                let mut out = tgt.clone();
                for dec in &self.decoders {
                    out = dec.forward(&out, &memory, Some(&causal), None);
                }
                let diff = out.sub_tensor(tgt);
                let mut loss = diff.pow_scalar(2.0).mean();
                if step == 0 || step + 1 == steps {
                    println!("AR  train step {}: loss={:.6}", step, loss.value());
                }
                loss.backward(None);
            }
            let mut params_step = self.parameters();
            opt.step(&mut params_step);
            opt.zero_grad(&mut params_step);
        }
    }

    fn triple(t: &Tensor) -> (usize, usize, usize) {
        let d = t.shape().dims();
        (d[0], d[1], d[2])
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Basic Transformer Example ===");

    let batch = 2usize;
    let src_len = 8usize;
    let tgt_len = 6usize;
    let embed = 32usize;
    let heads = 4usize;
    let layers = 2usize;

    let src = Tensor::randn(vec![batch, src_len, embed], Some(1001));
    let tgt = Tensor::randn(vec![batch, tgt_len, embed], Some(1002));

    let mut trf = BasicTransformer::new(embed, heads, layers, Some(999));
    let out = trf.forward(&src, &tgt);
    println!("Output shape: {:?}", out.shape().dims());

    // Quick optimization step
    let mut opt = Adam::with_learning_rate(0.005);
    let mut params = trf.parameters();
    for p in &params {
        opt.add_parameter(p);
    }
    let mut loss = out.mean();
    loss.backward(None);
    opt.step(&mut params);
    opt.zero_grad(&mut params);
    println!("Loss: {:.6}", loss.value());

    // Demo: non auto-regressive inference (single pass)
    let nar = trf.infer_non_autoregressive(&src, tgt_len);
    println!("NAR output shape: {:?}", nar.shape().dims());

    // Demo: auto-regressive inference (toy)
    let ar = trf.infer_autoregressive(&src, 3);
    println!("AR output shape: {:?}", ar.shape().dims());

    // NAR training demo
    let nar_tgt = tgt.clone();
    trf.train_non_autoregressive_steps(&src, &nar_tgt, 3, 0.01);

    // AR training demo (teacher-forced)
    let ar_tgt = tgt.clone();
    trf.train_autoregressive_steps(&src, &ar_tgt, 3, 0.01);
    println!("=== Done ===");
    Ok(())
}
