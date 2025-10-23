# Old code snippet of dynamic network pruning logic

```
def prune_node(self, node=None):

    if node == None:

        if len(self.nodes['hidden']) == 0:
            return

        node = np.random.choice(self.nodes['hidden'])

    if node in self.nodes['being pruned']:
        return

    self.nodes['being pruned'].append(node)

    for out_node in node.out_nodes.copy():
        self.prune_connection(node, out_node, node)

    for in_node in node.in_nodes.copy():
        self.prune_connection(in_node, node, node)

    for key in self.nodes:

        if key == 'layered':

            node_layer = find_sublist_index(node, self.nodes['layered'])
            self.nodes['layered'][node_layer].remove(node)

            if (node_layer != 0
                and node_layer != len(self.nodes['layered']) - 1): 
                if self.nodes['layered'][node_layer] == []:
                    self.nodes['layered'].remove(
                        self.nodes['layered'][node_layer])
        else:
            while node in self.nodes[key]:
                self.nodes[key].remove(node)   
        
def prune_connection(self,
                        in_node=None,
                        out_node=None,
                        calling_node=None):

    if in_node == None:

        if len(self.nodes['emitting']) == 0:
            return

        in_node = np.random.choice(self.nodes['emitting'])

    if out_node == None:

        out_node = np.random.choice(in_node.out_nodes)

    connection_was_already_pruned = in_node.disconnect_from(out_node)

    if connection_was_already_pruned:
        return
    
    self.nodes['receiving'].remove(out_node)
    self.nodes['emitting'].remove(in_node)

    if in_node != calling_node:

        if in_node not in self.nodes['emitting']:

            if in_node in self.nodes['input']:
                if self.input_net != None:
                    self.grow_connection(in_node=in_node)

            elif in_node in self.nodes['hidden']:
                self.prune_node(in_node)

    if out_node != calling_node:

        if out_node not in self.nodes['receiving']:

            if out_node in self.nodes['output']:
                if self.output_net != None:
                    self.grow_connection(out_node=out_node)

            elif out_node in self.nodes['hidden']:
                self.prune_node(out_node)

    for node in [in_node, out_node]:

        if (node != calling_node
            and node not in self.nodes['being pruned']):

            if node in self.nodes['hidden']:
                if node.in_nodes == [node] or node.out_nodes == [node]:
                    self.prune_node(node)

            elif node in self.nodes['output']:
                if node.in_nodes == [node] and node.out_nodes == [node]:
                    self.prune_connection(node, node)
```