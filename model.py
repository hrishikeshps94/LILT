from torch import nn
import torch
import torch.nn.functional as F
import torch.onnx.operators
from transformers.modeling_utils import apply_chunking_to_forward, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, TokenClassifierOutput
from transformers.activations import ACT2FN
from transformers.models.lilt import LiltConfig


class LiltTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # pad token id represents the padding token in the model
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size, padding_idx=config.pad_token_id)  # Maximum position embedding is the maximum number of tokens in the model 512+2([SEP],[CLS])
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)  # type_vocab_size is the number of token types in the model
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob, inplace=False)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(
            config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute")

    def create_position_ids_from_input_ids(self, input_ids, padding_idx):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1.
        Padding symbols are ignored. This is modified from fairseq's `utils.make_positions`.
        Args:
            input_ids: torch.Tensor
            padding_idx: int
        Returns:
            torch.Tensor
        Examples:
            >>> input_ids = torch.tensor([[1, 2, 3, 0, 1]])
            >>> create_position_ids_from_input_ids(input_ids, 0)
            tensor([[1, 2, 3, 0, 4]])
        """
        # Create a mask where non-padding tokens are represented by 1 and padding tokens by 0
        # e.g., tensor([[1, 1, 1, 0, 1]])
        mask = input_ids.ne(padding_idx).int()

        # Compute the cumulative sum of the mask elements along the sequence dimension (dim=1)
        # and add the length of past key values to it
        incremental_indices = torch.cumsum(mask, dim=1).type_as(
            mask)  # e.g., tensor([[1, 2, 3, 3, 4]])

        # Multiply the incremental indices by the mask and add the padding index
        # This results in a tensor where each non-padding token is replaced by its position number (starting from padding_idx+1),
        # and each padding token is replaced by padding_idx
        # e.g., tensor([[1, 2, 3, 0, 4]])
        return (incremental_indices * mask).long() + padding_idx

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None):

        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids, self.config.pad_token_id).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_input_embeds(
                    inputs_embeds)
        if input_ids is not None:
            input_shape = input_ids.size()

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long).to(input_ids.device)

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, position_ids


class LiltLayoutEmbeddings(nn.Module):
    """
    A PyTorch module for generating embeddings for bounding boxes in a 2D space.

    Args:
        config: A configuration object with the necessary parameters for the embeddings.

    Example:
        >>> config = Config(max_2d_position_embeddings=1000, hidden_size=768, channel_shrink_ratio=6, pad_token_id=0, layer_norm_eps=1e-12, hidden_dropout_prob=0.1)
        >>> embeddings_module = LiltLayoutEmbeddings(config)
        # 2 bounding boxes
        >>> bbox = torch.tensor([[[0, 0, 10, 10], [10, 10, 20, 20]]])
        >>> position_ids = torch.tensor([[1, 2]])  # Position ids for the boxes
        >>> embeddings = embeddings_module(bbox, position_ids)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Create embeddings for x, y, height, width, and box position
        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size//6)
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size//6)
        self.h_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size//6)
        self.w_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size//6)
        self.box_position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size//self.config.channel_shrink_ratio, padding_idx=config.pad_token_id)
        # Linear layer for transforming the box embeddings
        self.box_linear_embeddings = nn.Linear(
            config.hidden_size, config.hidden_size//self.config.channel_shrink_ratio)
        # Layer normalization and dropout for regularization
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size//self.config.channel_shrink_ratio, eps=config.layer_norm_eps, elementwise_affine=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob, inplace=False)

    def forward(self, bbox=None, position_ids=None):
        """
        Generate embeddings for the bounding boxes and their positions.

        Args:
            bbox: A tensor of shape (batch_size, num_boxes, 4) representing the bounding boxes.
            position_ids: A tensor of shape (batch_size, num_boxes) representing the position ids of the boxes.

        Returns:
            A tensor of shape (batch_size, num_boxes, hidden_size) representing the embeddings.
        """
        try:
            # Extract the left, top, right, and bottom coordinates of each bounding box
            left_position_embeddings = self.x_position_embeddings(
                bbox[:, :, 0])
            top_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(
                bbox[:, :, 2])
            bottom_position_embeddings = self.y_position_embeddings(
                bbox[:, :, 3])
        except IndexError as e:
            raise IndexError(
                "Bounding box must have 4 coordinates (left, top, right, bottom).") from e
        # Compute the height and width of each bounding box
        h_position_embeddings = self.h_position_embeddings(
            bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(
            bbox[:, :, 2] - bbox[:, :, 0])
        # Concatenate all the embeddings along the last dimension
        spatial_embeddings = torch.cat([left_position_embeddings, top_position_embeddings,
                                        right_position_embeddings, bottom_position_embeddings,
                                        h_position_embeddings, w_position_embeddings], dim=-1)
        # Transform the spatial embeddings
        spatial_embeddings = self.box_linear_embeddings(spatial_embeddings)
        # Generate embeddings for the box positions
        bbox_position_embeddings = self.box_position_embeddings(position_ids)
        # Add the spatial embeddings and the box position embeddings
        embeddings = spatial_embeddings + bbox_position_embeddings
        # Normalize and regularize the embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LiltSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.attention_head_size = self.config.hidden_size // self.config.num_attention_heads  # 768/12=64
        self.all_head_size = self.config.num_attention_heads * \
            self.attention_head_size  # 12*64=768
        # Query, key, and value linear transformations for text embeddings
        self.query = nn.Linear(self.config.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.config.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.config.hidden_size, self.all_head_size)

        # Query, key, and value linear transformations for layout embeddings
        self.layout_query = nn.Linear(
            self.config.hidden_size // self.config.channel_shrink_ratio, self.all_head_size // self.config.channel_shrink_ratio)
        self.layout_key = nn.Linear(
            self.config.hidden_size // self.config.channel_shrink_ratio, self.all_head_size // self.config.channel_shrink_ratio)
        self.layout_value = nn.Linear(
            self.config.hidden_size // self.config.channel_shrink_ratio, self.all_head_size // self.config.channel_shrink_ratio)

        self.dropout = nn.Dropout(self.config.attention_probs_dropout_prob)

    def forward(self, hidden_states, layout_inputs, attention_mask=None, head_mask=None, output_attentions=None):
        """ Apply self-attention mechanism to the input hidden states.
        """
        # Compute the query, key, and value tensors for text embeddings
        Lay_Batch, Lay_Seq_Len, Lay_Hidden_Size = layout_inputs.size()
        Hid_Batch, Hid_Seq_Len, Hid_Hidden_Size = hidden_states.size()

        layout_value = self.layout_value(layout_inputs).view(
            Lay_Batch, Lay_Seq_Len, self.config.num_attention_heads, self.attention_head_size//self.config.channel_shrink_ratio).transpose(1, 2)
        layout_key = self.layout_key(layout_inputs).view(
            Lay_Batch, Lay_Seq_Len, self.config.num_attention_heads, self.attention_head_size//self.config.channel_shrink_ratio).transpose(1, 2)
        layout_query = self.layout_query(layout_inputs).view(
            Lay_Batch, Lay_Seq_Len, self.config.num_attention_heads, self.attention_head_size//self.config.channel_shrink_ratio).transpose(1, 2)

        key_layer = self.key(hidden_states).view(
            Hid_Batch, Hid_Seq_Len, self.config.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value_layer = self.value(hidden_states).view(
            Hid_Batch, Hid_Seq_Len, self.config.num_attention_heads, self.attention_head_size).transpose(1, 2)

        query_layer = self.query(hidden_states).view(
            Hid_Batch, Hid_Seq_Len, self.config.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # Flash attention Implementation tried but summition of attention scores is not possible
        # if (attention_mask is None) and (head_mask is not None):
        #     layout_context_layer = F.scaled_dot_product_attention(
        #         layout_query, layout_key, layout_value, head_mask, self.config.attention_probs_dropout_prob)
        #     context_layer = F.scaled_dot_product_attention(
        #         query_layer, key_layer, value_layer, head_mask, self.config.attention_probs_dropout_prob)
        # else:
        attention_scores = query_layer @ key_layer.transpose(-1, -2)
        layout_attention_scores = layout_query @ layout_key.transpose(
            -1, -2)
        tmp_attention_scores = attention_scores / self.attention_head_size**0.5
        tmp_layout_attention_scores = layout_attention_scores / \
            (self.attention_head_size//self.config.channel_shrink_ratio)**0.5
        attention_scores = tmp_attention_scores + tmp_layout_attention_scores
        layout_attention_scores = tmp_layout_attention_scores + tmp_attention_scores

        if attention_mask is not None:
            # Apply the attention mask to the attention scores
            attention_scores = attention_scores + attention_mask
            layout_attention_scores = layout_attention_scores + attention_mask

        # Normalize the attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        layout_attention_probs = F.softmax(layout_attention_scores, dim=-1)

        # Apply dropout to the attention probabilities
        attention_probs = self.dropout(attention_probs)
        layout_attention_probs = self.dropout(layout_attention_probs)

        if head_mask is not None:
            # Apply the head mask to the attention probabilities
            attention_probs = attention_probs * head_mask
            layout_attention_probs = layout_attention_probs * head_mask

        layout_context_layer = layout_attention_probs @ layout_value
        context_layer = attention_probs @ value_layer

        layout_context_layer = layout_context_layer.transpose(
            1, 2).contiguous().view(Lay_Batch, Lay_Seq_Len, self.all_head_size//self.config.channel_shrink_ratio)
        context_layer = context_layer.transpose(
            1, 2).contiguous().view(Hid_Batch, Hid_Seq_Len, self.all_head_size)

        outputs = ((context_layer, layout_context_layer), attention_probs) if output_attentions else (
            (context_layer, layout_context_layer),)

        return outputs


class LiltSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob, inplace=False)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LiltAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self = LiltSelfAttention(config)
        self.output = LiltSelfOutput(config)
        self.pruned_heads = set()

        ori_hidden_size = config.hidden_size
        config.hidden_size = config.hidden_size // config.channel_shrink_ratio
        self.layout_output = LiltSelfOutput(config)
        config.hidden_size = ori_hidden_size

    def forward(self, hidden_states, layout_inputs, attention_mask=None, head_mask=None, output_attentions=None):
        self_outputs = self.self(hidden_states, layout_inputs, attention_mask,
                                 head_mask, output_attentions)
        attention_output = self.output(self_outputs[0][0], hidden_states)
        layout_attention_output = self.layout_output(
            self_outputs[0][1], layout_inputs)
        # add attentions if we output them
        # add attentions if we output them
        return ((attention_output, layout_attention_output),) + self_outputs[1:]


class LiltIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(
            config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LiltOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob, inplace=False)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LiltLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # default 0 this will come as part of pretrained config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LiltAttention(config)
        self.intermediate = LiltIntermediate(config)
        self.output = LiltOutput(config)

        ori_hidden_size = config.hidden_size
        ori_intermediate_size = config.intermediate_size
        config.hidden_size = config.hidden_size // config.channel_shrink_ratio
        config.intermediate_size = config.intermediate_size // config.channel_shrink_ratio
        self.layout_intermediate = LiltIntermediate(config)
        self.layout_output = LiltOutput(config)
        config.hidden_size = ori_hidden_size
        config.intermediate_size = ori_intermediate_size

    def forward(self, hiddens_states, layout_inputs, attention_mask=None,
                head_mask=None, output_attentions=None):
        self_attention_outputs = self.attention(hiddens_states, layout_inputs, attention_mask,
                                                head_mask, output_attentions)
        attention_output = self_attention_outputs[0][0]
        layout_attention_output = self_attention_outputs[0][1]

        # add self_attention_outputs if we output attention weights
        outputs = self_attention_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)

        layout_layer_output = apply_chunking_to_forward(
            self.layout_feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, layout_attention_output)

        outputs = ((layer_output, layout_layer_output),) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def layout_feed_forward_chunk(self, layout_attention_output):
        layout_intermediate_output = self.layout_intermediate(
            layout_attention_output)
        layout_layer_output = self.layout_output(
            layout_intermediate_output, layout_attention_output)
        return layout_layer_output


class LiltEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([LiltLayer(config)
                                   for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, layout_inputs, attention_mask=None, head_mask=None, output_attentions=None, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_attentions else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(hidden_states, layout_inputs, attention_mask,
                                         layer_head_mask, output_attentions)
            hidden_states = layer_outputs[0][0]
            layout_inputs = layer_outputs[0][1]

            if output_attentions:
                all_self_attentions = all_self_attentions + \
                    (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None), layer_outputs

        return BaseModelOutput(last_hidden_state=hidden_states,
                               hidden_states=all_hidden_states, attentions=all_self_attentions), layer_outputs


# Copied from transformers.models.bert.modeling_bert.BertPooler
class LiltPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LiltPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LiltConfig
    base_model_prefix = "lilt"
    _no_split_modules = []

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class CustomLiLTModel(LiltPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = LiltTextEmbeddings(config)
        self.layout_embeddings = LiltLayoutEmbeddings(config)

        self.encoder = LiltEncoder(config)

        self.pooler = LiltPooler(config) if add_pooling_layer else None

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError(
                "You have to specify either input_ids")

        device = input_ids.device if input_ids is not None else torch.device(
            "cpu")

        if attention_mask is None:
            attention_mask = torch.ones_like(
                input_ids, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(
                input_ids, dtype=torch.long, device=device)

        if bbox is None:
            raise ValueError(
                "You have to specify bbox")
        # input_shape = input_ids.size()

        # extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
        #     attention_mask, input_shape, device)  # Remove device from here ,going to be deprecated
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (
            1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output, position_ids = self.embeddings(input_ids=input_ids, position_ids=position_ids,
                                                         token_type_ids=token_type_ids)

        layout_embedding_output = self.layout_embeddings(
            bbox=bbox, position_ids=position_ids)

        encoder_outputs, layout_encoder_outputs = self.encoder(embedding_output, layout_embedding_output, attention_mask=extended_attention_mask,
                                                               head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(
            sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(last_hidden_state=sequence_output,
                                          pooler_output=pooled_output,
                                          hidden_states=encoder_outputs.hidden_states,
                                          attentions=encoder_outputs.attentions)


class TokenClassificationModel(LiltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.lilt = CustomLiLTModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, bbox=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=True):
        lilt_output = self.lilt(input_ids, bbox, attention_mask, token_type_ids,
                                position_ids, head_mask, output_attentions, output_hidden_states, return_dict)
        sequence_output = lilt_output[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None

        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1))
        if not return_dict:
            output = (logits,) + lilt_output[2:]
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(loss=loss,
                                     logits=logits,
                                     hidden_states=lilt_output.hidden_states,
                                     attentions=lilt_output.attentions)
