/**
* Returns a lower triangular matrix wish shape (seqLen, seqLen)
*
*/
function lookAheadMask(seqLen){
const inputOnes = tf.ones([seqLen, seqLen]);
const mask = tf.linalg.bandPart(inputOnes, -1, 0);
return mask;

}


/** Return a 3D tf.tensor, which is the algebraic manipulation of the decoder embedding  
* Args
*  decoder_embedded: 3d tf.tensor 
*  dec_size: int32, decoder size
*  seq_len: int32
*  posEnc: 3d tf.tensor, position encoding  
*/
function decodeEmbedNew(decoder_embedded, dec_size, seq_len, posEnc){
  const decoder_embedded_new = decoder_embedded.mul(tf.sqrt(tf.cast(dec_size,'float32')));
  const output = decoder_embedded_new.add(posEnc.slice([0, 0, 0], [-1, seq_len, -1]))
  console.log(output.print());
  return output;

}


/** Manipulate the shape of input tensor x
* 
* Args
* x: tf.tensor, shape = [batch_size, time_steps, dec_size]
* batch_size, dec_size, num_heads: int32
* return: tf.tensor, shape = [batch_size, num_heads, time_steps, dec_size]
*/
function splitHeads(x, batch_size, dec_size, num_heads){
var depth = Math.floor(dec_size/num_heads);
var output = x.reshape([batch_size, -1, num_heads, depth]);

return tf.transpose(output, [0, 2, 1, 3]);


/** Calculate the attention weights
* 
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
   * seq_len_k == seq_len_v
  Returns:
    list: [output, attention_weights] 
*/
function scaledDotProductAttention(q, k, v, mask){
  var matmulQk = q.matMul(k, transposeA=false, transposeB=true);
  
  // scale matmul_qk
  var kShape = k.shape;
  var dk = tf.cast(kShape[kShape.length-1], 'float32');
  var scaledAttentionLogits = matmulQk.div(tf.sqrt(dk));

  // add the mask to the scaled tensor.
  if (mask != null){
    scaledAttentionLogits = scaledAttentionLogits.add(mask.mul(tf.scalar(-1e9))); }
     
  // softmax is normalized on the last axis (seq_len_k) so that the scores
  // add up to 1.
  var attentionWeights = scaledAttentionLogits.softmax();  // (..., seq_len_q, seq_len_k)
  // console.log(attentionWeights.shape); // [3,5,4] 
  

  var output = tf.matMul(attentionWeights, v); // (..., seq_len_q, depth_v)

  return [output, attentionWeights];


}}