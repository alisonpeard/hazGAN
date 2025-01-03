"""
Algotithm:
1. Train critic
    a. generator.trainable_weights += (generator.trainable_weights - generator._last_weights)
    b. update critic -> critic.trainable_weights
    c. restore generator.trainable_weights
    d. save generator.trainable_weights as generator._last_weights 
2. Train generator
    a. critic += (critic.trainable_weights - critic._last_weights)
    b. update generator -> generator.trainable_weights
    c. restore critic.trainable_weights
    d. save critic.trainable_weights as critic._last_weights
"""

__all__ = ["lookahead"]


def lookahead(model_name):
    def decorator(func):
        def wrapper(self, *args, **kwargs):

            model = getattr(self, model_name)
            original_weights = list(model.trainable_weights)

            if hasattr(model, "_last_weights"):
                for i, weight in enumerate(model.trainable_weights):
                    delta_w = weight - model._last_weights[i]
                    weight.assign(weight + delta_w)
                    
            result = func(self, *args, **kwargs)

            for i, weight in enumerate(model.trainable_weights):
                weight.assign(original_weights[i])
            
            model._last_weights = list(model.trainable_weights)
            
            return result
        return wrapper
    return decorator