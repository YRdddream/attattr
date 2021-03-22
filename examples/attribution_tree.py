import numpy as np
import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attr_file",
                        default=None,
                        type=str,
                        help="The file of attribution scores.")
    parser.add_argument("--tokens_file",
                        default=None,
                        type=str,
                        help="The file that contains tokens of the target example.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--example_index",
                        default=128,
                        type=int,
                        help="Example index.")
    args = parser.parse_args()

    task_name = args.task_name
    example_index = args.example_index

    with open(args.attr_file) as fin:
        att_all = []
        for line in fin:
            att_all.append(json.loads(line))
        att_all = np.array(att_all).sum(1)
    proportion_all = copy.deepcopy(att_all)
    for i in range(len(proportion_all)):
        proportion_all[i] /= abs(proportion_all[i][1:,:].max())

    # adjust the threshold
    threshold = 0.4
    proportion_all *= (proportion_all > threshold).astype(int) 

    seq_length = len(proportion_all[0])
    height_list = [0 for i in range(seq_length)]
    # -1: not appear  0: appear but not fixed  1: fixed   
    fixed_list = [-1 for i in range(seq_length)]   
    edges = []

    # find the top node
    ig_remain = [0 for i in range(seq_length)]
    att_combine_layer = att_all.sum(0) / abs(att_all.sum(0).max())
    att_combine_layer *= (1 - np.identity(len(att_combine_layer))) * (att_combine_layer > 0)
    att_combine_layer[0] *= 0
    arg_res = np.argsort(att_combine_layer.sum(-1))[::-1]

    top_token_index = arg_res[0] if arg_res[0] != 0 else arg_res[1]
    height_list[top_token_index] = 11/12
    fixed_list[top_token_index] = 0
    for i in range(seq_length):
        if i != top_token_index and proportion_all[10][top_token_index][i] > threshold:
            fixed_list[i] = 0
            fixed_list[top_token_index] = 1
            edges.append((top_token_index, i))

    for layer_index in range(9, -1, -1):
        for i_token in range(1, seq_length):
            for j_token in range(0, seq_length):
                if proportion_all[layer_index][i_token][j_token] < threshold or fixed_list[i_token] == -1:
                    continue
                if fixed_list[j_token] == 1:
                    continue
                if (i_token, j_token) in edges:
                    continue
                if fixed_list[i_token] == 0 and fixed_list[j_token] == 0:
                    continue
                if fixed_list[i_token] == 1 and fixed_list[j_token] == 0:  
                    continue
                if fixed_list[i_token] == 0 and fixed_list[j_token] == -1:
                    fixed_list[i_token] = 1
                    fixed_list[j_token] = 0
                    height_list[j_token] = ((height_list[i_token]) * 12 - 1) / 12
                if fixed_list[i_token] == 1 and fixed_list[j_token] == -1:
                    fixed_list[j_token] = 0
                    height_list[j_token] = min(height_list)
                edges.append((i_token, j_token))

    # token examples
    # tokens = ["[CLS]", "i", "don", "'", "t", "know", "um", "do", "you", "do", "a", "lot", "of", "camping", "[SEP]", "I", "know", "exactly", ".", "[SEP]"]
    # tokens = ["[CLS]", "The", "new", "rights", "are", "nice", "enough", "[SEP]", "Everyone", "really", "likes", "the", "newest", "benefits", "[SEP]"]
    # tokens = ["[CLS]", "so", "i", "have", "to", "find", "a", "way", "to", "supplement", "that", "[SEP]", "I", "need", "a", "way", "to", "add", "something", "extra", ".", "[SEP]"]
    with open(args.tokens_file) as fin:
        tokens_all = json.load(fin)
    tokens = tokens_all[example_index]["tokens"]

    for i in range(len(tokens)):
        tokens[i] = tokens[i] + str(i)

    fig1 = plt.figure(1,figsize=(30,22)) 
    fig1.patch.set_facecolor('xkcd:white')

    G = nx.DiGraph()

    for token in tokens:
        G.add_node(token)

    for (i_token, j_token) in edges:
        G.add_weighted_edges_from([(tokens[i_token], tokens[j_token], 0.5)])

    fix_position = {tokens[i]:[i/len(tokens), height_list[i]] for i in range(len(tokens))} 
    M = G.number_of_edges()
    pos = nx.spring_layout(G, pos=fix_position)
    edge_colors = range(2, M + 2)

    unused_node = list(nx.isolates(G))
    for node in unused_node:
        G.remove_node(node)

    edge_alphas = []
    edges_list = list(G.edges.data())
    for single_edge in edges_list:
        edge_alphas.append(single_edge[2]["weight"])

    nodes = nx.draw_networkx_nodes(G, fix_position, node_size=700, node_color='blue')
    edges = nx.draw_networkx_edges(G, fix_position, node_size=700, arrowstyle='->',
                                arrowsize=10, edge_color='b',
                                edge_cmap=plt.cm.Blues, width=2)
    nx.draw_networkx_labels(G, fix_position, font_size=15, font_family='sans-serif')

    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    pc.set_array(edge_colors)

    ax = plt.gca()
    ax.set_axis_off()
    plt.show()


if __name__ == "__main__":
    main()