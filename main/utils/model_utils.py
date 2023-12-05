import torch
from main.models.model import ClassfierModel


def load_model(model_file, use_gpu=False):
    # load the pre-trained model
    with open(model_file, 'rb') as f:
        # If we want to use GPU and CUDA is correctly installed
        if use_gpu and torch.cuda.is_available():
            state = torch.load(f)
        else:
            # Load all tensors onto the CPU
            state = torch.load(f, map_location='cpu')
    if not state:
        return None
    vocab_dict = state['vocab_dict']
    index2label = state['index2label']
    config = {'hidden_size': state['hidden_size'], 'dropout': state['dropout']}

    model_classifier = ClassfierModel(vocab_dict=vocab_dict, index2label=index2label, config=config)
    model_classifier.encoder.load_state_dict(state['state_dict'])

    if use_gpu:
        model_classifier.encoder.cuda()

    return model_classifier
