# The core code of adversarial distribution learning. Complete code will be released soon.     
    for i in range(args.num_iters):
        optimizer.zero_grad()
        coeffs = F.gumbel_softmax(log_coeffs.unsqueeze(0).repeat(args.batch_size, 1, 1), hard=False) # B x T x V  Gumbel-Softmax
        # coeffs = F.gumbel_softmax(log_coeffs.unsqueeze(0).repeat(args.batch_size, 1, 1), hard=True) # B x T x V  ST Gumbel-Softmax
        inputs_embeds = (coeffs @ embeddings[None, :, :]) # B x T x D
        pred = model(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids_batch).logits
        
        top_preds = pred.sort(descending=True)[1]
        correct = (top_preds[:, 0] == label).long()
        indices = top_preds.gather(1, correct.view(-1, 1))
        adv_loss = (pred[:, label] - pred.gather(1, indices) + args.kappa).clamp(min=0).mean()
        
        # Similarity constraint
        ref_embeds = (coeffs @ ref_embeddings[None, :, :])
        pred = ref_model(inputs_embeds=ref_embeds)
        
        output = pred.hidden_states[args.embed_layer]  
        if args.model == 'gpt2' or 'bert-base-uncased' or 'roberta-base' or 'albert-base-v2' or 'distilbert-base-uncased' in args.model:
            output = output[:, -1]
        else:
            output = output.mean(1)
        cosine = (output * orig_output).sum(1) / output.norm(2, 1) / orig_output.norm(2, 1)
        ref_loss = 1 - (args.lam_sim * cosine.mean())
        perp_loss = args.lam_perp * log_perplexity(pred.logits, coeffs)
            
        # Compute loss and backward
        if adv_loss == 0:
            total_loss =  1/2.0*torch.log(abs(perp_loss+ref_loss))
        else:
            total_loss = torch.log(abs(adv_loss+1)) + 1/2.0*torch.log(abs(perp_loss+ref_loss))
        total_loss.backward()
            


