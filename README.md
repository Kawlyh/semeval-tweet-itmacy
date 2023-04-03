# semeval-tweet-itmacy
use 6-different-language tweet data to predict itmacy
and the last score is made on 4-different-language tweet data(not in the training data)
# in this project,we use:
1.pretrain model to transfer knowledge to this model(xmlroberta)
2.use back-traslation to augment training data
3. use label smooth to add noise to the model
4. use r-drop to enhance the score
