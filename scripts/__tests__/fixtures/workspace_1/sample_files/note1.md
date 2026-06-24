# MoE routing note

The Granite MoE coordinator selects experts by argmax over the router logits
at layer 5. Top-k experts are dispatched; their outputs are recombined
downstream before the lm_head.
