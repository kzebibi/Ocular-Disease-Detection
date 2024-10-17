from models.custom_model import create_custom_model
from models.densenet121_model import create_densenet121_model
from models.xception_model import create_xception_model

def create_model(model_name, config):
    model_config = config['models'][model_name]
    input_shape = tuple(model_config['input_shape'])
    num_classes = model_config['num_classes']

    if model_name.lower() == 'densenet121':
        model = create_densenet121_model(input_shape, num_classes)
    elif model_name.lower() == 'xception':
        model = create_xception_model(input_shape, num_classes)
    elif model_name.lower() == 'custom':
        model = create_custom_model(input_shape, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model

def compile_model(model, model_name, config):
    model_config = config['models'][model_name]
    learning_rate = model_config.get('learning_rate', 0.001)

    if model_name.lower() == 'densenet121':
        from models.densenet121_model import compile_densenet121_model
        return compile_densenet121_model(model, lr=learning_rate)
    elif model_name.lower() == 'xception':
        from models.xception_model import compile_xception_model
        return compile_xception_model(model, lr=learning_rate)
    elif model_name.lower() == 'custom':
        from models.custom_model import compile_custom_model
        return compile_custom_model(model)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
