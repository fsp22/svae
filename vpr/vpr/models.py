import torch


def kld(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return eps.mul(std).add_(mu)


class EncoderDecoder(torch.nn.Module):
    def __init__(self, p_dims):
        super(EncoderDecoder, self).__init__()
        self.p_dims = p_dims

        self.p_layers = torch.nn.ModuleList([torch.nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])

        # for layer in self.p_layers:
        #     nn.init.xavier_normal(layer.weight)

        self.activation = torch.nn.ReLU()

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = self.activation(h)
        return h


class VPR(torch.nn.Module):
    def __init__(self, n_items, hp):
        super(VPR, self).__init__()

        self.hp = hp
        self.embed_item = torch.nn.Embedding(n_items + 1, hp['embed_size'])

        self.encoder = EncoderDecoder([n_items, hp['hidden_size_enc'], 2 * hp['embed_size']])

        for m in self.modules():
            if isinstance(m, torch.nn.Embedding) or isinstance(m, torch.nn.Linear):
                # torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.xavier_normal(m.weight)

    def forward(self, user_items, item_i, item_j=None):
        """

        :param user_items: bs x n_items
        :param item_i:     bs x k_items
        :param item_j:     bs x k_items
        :return:    bs x k_items
                    bs x k_items, bs (mu), bs (logvar)
        """
        mask = item_i > 0
        item_i = self.embed_item(item_i)           # BS x k_items x embed_size

        h = self.encoder(user_items)
        mu = h[:, :self.hp['embed_size']]
        logvar = h[:, self.hp['embed_size']:]

        user = reparameterize(mu, logvar)          # bs x embed_size
        k_items = item_i.shape[1]
        user = user.unsqueeze(1).repeat(1, k_items, 1)

        if item_j is not None:
            item_j = self.embed_item(item_j)       # BS x k_items x embed_size

            score = (user * (item_i - item_j)).sum(-1)

            return score, mask, mu, logvar
        else:
            prediction_i = (user * item_i).sum(-1)
            return prediction_i

    def score(self, user_items, items):
        """
        Compute the score for each item in items for a user
        :param user:     bs x n_items
        :param items:    bs x (pos + neg)
        :return: n_items
        """
        items = self.embed_item(items)               # bs x n_items x emb_size

        h = self.encoder(user_items)

        # we assume that here mu is z
        user = h[:, :self.hp['embed_size']]         # bs x emb_size

        user_expanded = user.unsqueeze(1).repeat(1, items.shape[1], 1)   # bs x n_items x emb_size
        scores = (user_expanded * items).sum(-1)
        return scores
