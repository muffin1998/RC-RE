import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from transformers import TFAutoModel


class RCBert(Model):

    def __init__(self, **kwargs):
        super(RCBert, self).__init__()
        self.bert_model = kwargs.pop('bert_model', 'bert-base-uncased')
        self.hidden_state_unit = kwargs.pop('hidden_state_unit')
        self.dropout_rate = kwargs.pop('dropout_rate', 0.1)
        self.num_class = kwargs.pop('num_class')
        self.r_num = kwargs.pop('num_relation')
        self.pos_weight = kwargs.pop('pos_weight', 1.0)

        self.bert = TFAutoModel.from_pretrained(self.bert_model)
        self.dropout = Dropout(self.dropout_rate)
        self.rel_embedding = self.add_weight(shape=(self.r_num, self.hidden_state_unit), name='rel_emb')
        self.classifier = Dense(self.num_class, activation='softmax')
        self.b_classifier = Dense(1)
        self.rel_dense = Dense(self.hidden_state_unit)
        self.att_dense = Dense(self.hidden_state_unit)
        self.cls_dense = Dense(self.hidden_state_unit)
        self.ent_dense = Dense(self.hidden_state_unit)
        self.output_dense = Dense(4 * self.hidden_state_unit)

    def call(self, inputs, training=None, mask=None):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        e1_mask = inputs['e1_mask']
        e2_mask = inputs['e2_mask']
        r_label = inputs['r_label']

        (seq_outputs, _) = self.bert(input_ids, attention_mask=attention_mask, training=training,
                                     return_dict=False, output_hidden_states=False)

        def dropout_and_activate(features):
            return self.dropout(tf.nn.tanh(features), training=training)

        def entity_output(s_outputs, entity_mask):
            entity_mask = tf.expand_dims(tf.cast(entity_mask, dtype='float32'), axis=-1)
            return tf.reduce_sum(tf.multiply(s_outputs, entity_mask), axis=1) / tf.reduce_sum(entity_mask, axis=1)

        cls_outputs = seq_outputs[:, 0]
        e1_outputs = entity_output(seq_outputs, e1_mask)
        e2_outputs = entity_output(seq_outputs, e2_mask)

        cls_outputs = self.cls_dense(dropout_and_activate(cls_outputs))
        e1_outputs = self.ent_dense(dropout_and_activate(e1_outputs))
        e2_outputs = self.ent_dense(dropout_and_activate(e2_outputs))

        # relation-attended
        r_query = tf.broadcast_to(self.rel_embedding[tf.newaxis, :, :],
                                  shape=(seq_outputs.shape[0], *self.rel_embedding.shape))
        r_scores = tf.matmul(r_query, seq_outputs, transpose_b=True)
        r_scores = tf.nn.softmax(r_scores, axis=-1)
        r_key = tf.expand_dims(seq_outputs, 1)
        r_outputs = tf.reduce_sum(tf.multiply(tf.expand_dims(r_scores, axis=-1), r_key),
                                  axis=-2)
        att_outputs = tf.reduce_max(r_outputs, axis=1)
        att_outputs = self.att_dense(dropout_and_activate(att_outputs))

        # binary loss
        r_outputs = tf.reshape(r_outputs, shape=(-1, r_outputs.shape[-1]))
        r_logits = self.b_classifier(r_outputs)
        r_true = tf.one_hot(r_label, depth=self.r_num)
        r_true = tf.reshape(r_true, shape=(-1, 1))
        bce_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(r_true, r_logits, self.pos_weight))

        final_outputs = tf.concat([att_outputs, e1_outputs, e2_outputs, cls_outputs], axis=-1)
        final_outputs = self.dropout(final_outputs, training=training)
        predictions = self.classifier(final_outputs)

        return predictions, bce_loss

