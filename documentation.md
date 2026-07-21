# Summary
The model is a message-passing graph neural network.
## Local environment
The local environment representation is built using features 1-6+11 of the provided atomic embeddings. These go through a dense layer to a higher dimensionality learned embedding. Edge representations start from the first three dimensions of the provided edge embeddings - disregarding the aromatic dimension. These also go through a dense layer to a higher dimensionality learned embedding, which is concatenated with an RBF expansion of the edge length. Graph level attributes will be initialised as all 1s then updated through the message-passing layers. These will be the default settings, changeable via the config.json.
## Message-passing layers
Node, edge, and graph-level attributes are updated using N message passing layers.
The Battaglia GNN layer equations, updates edge attributes $\vec{e}$ to those in the next layer $\vec{e}^{'}$ using update functions $\phi$:  
$\vec{e}^{'}_k = \phi^e\left( \vec{e}_k, \vec{v}_{rk}, \vec{v}_{sk}, \vec{u}\right),$  
and then node attributes $\vec{v}$:   
$\vec{v}^{'}_i = \phi^v\left(\bar{e}_{i}^{'},\vec{v}_i,\vec{u}\right)$  
and finally graph level attributes $\vec{u}$:  
$\vec{u}^{'}=\phi^{u}(\bar{e}^{'},\bar{v}^{'},\vec{u}),$  
where we use aggregate functions $\rho$ over the sets of nodes $V$ and edges $E$:  
$\bar{e}_i^{'}=\rho^{e\to v}(E_{i}^{'}),$  
$\bar{e}^{'}=\rho^{e\to u}(E^{'}),$  
and  
$\bar{v}^{'}=\rho^{v\to u}(V^{'}).$  
The architecture used here is similar to MEGNet, which goes beyond SchNet's use of message-passing layers (effectively - though they don't explicitly use graphs but treat them as convolutional filters) at the node level, going up to edge and graph level. Many similar models have been used as MLIPs, for which the energy is explicitly decomposed as a sum of local contributions. However, HOMO-LUMO gap is an intensive, graph-level property. Intensive pooling (e.g. mean) could be used after message passing only on node (and optionally edge) attributes, but here I wanted to see how well aggregating up to the graph level works. MEGNet's architecture is rather complicated, using e.g. set2set layers, but this model will take a more minimal approach to a full GNN.  
For the update functions, I will use two-layer MLPs with SiLU activation after the first. This is simpler than the approaches taken e.g. in SchNet and CGCNN but should be reasonable.  
This architecture ensures permutation, translation, and rotation invariance as required for this property, and can distinguish between two molecules with the same composition but marginally different bond lengths. However, three-body interactions (e.g. bond angles) aren't properly dealt with. They are indirectly dealt with by message passing, but there could be isomers with different bond angles that this model would incorrectly predict as having identical HOMO-LUMO gaps. However, this approach should still perform well - this will only subtly change the gaps. Solutions exist e.g. ALIGNN uses a second graph where the bonds are the nodes and then third-order interactions are the edges.
## Prediction head
This is a simple MLP with two hidden layers and SiLU activation, with the second hidden layer having half as many dims as the first, and then outputting the scalar target.

## Implementation choices
AdamW optimizer, 128 batch size, 150 epochs, OneCycleLR - all fairly standard choices aiming to complete training quite quickly. Training completed in about 25 mins on a NVIDIA GeForce RTX 5090 GPU. Used 4 message-passing layers. Too few and atoms far apart won't interact, though including the graph-level attributes should reduce the number of layers needed in this respect. SchNet used 6 layers for their best performance on QM9 but only update node attributes. Too many layers can lead to over-smoothing, so 4 seemed like a reasonable starting point.

## Performance
When trained as outlined above, best validation MAE was 0.068 eV - see evaluation notebook for loss curve and prediction vs true plot. This is a little higher than similar models like MEGNet and SchNet, with MAEs 0.061 and 0.063 eV, respectively (on the same size test set, though not necessarily the same molecules). However, hyperparameter tuning would likely bring performance much closer to those. An initial trial with 3 blocks and 256 batch size gave 0.086 eV, so going to smaller batches/more epochs and more message passing layers may further improve performance. The model is quite lighweight, with 601k trainable parameters when using the settings in the config. Just over 5 parameters per training example is quite low for chemical ML models, especially with heavy inductive bias as we have here. Thus, we would likely get better performance by increasing e.g. the feature, edge, and global embedding lengths. The loss curve in the notebook shows plateauing but perhaps we could train a little longer - early stopping with a patience of 15 epochs was not triggered. With more time/resources I would tune the hyperparameters of the model, train for longer, and explicitly encode third-order interactions into the architecture like is done in M3GNet or ALIGNN.
