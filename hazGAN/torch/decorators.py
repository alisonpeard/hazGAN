import torch



def lookahead(model_name):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            model = getattr(self, model_name)
            
            # Store original weights
            original_weights = [tf.identity(w) for w in model.trainable_weights]
            
            try:
                # Perform lookahead modifications
                if hasattr(self, 'generator_wake'):
                    for i, weight in enumerate(model.trainable_weights):
                        delta_w = weight - self.generator_wake[i]
                        weight.assign(weight + delta_w)
                
                # Call the original function
                result = func(self, *args, **kwargs)
                
            finally:
                # Restore original weights
                for i, weight in enumerate(model.trainable_weights):
                    weight.assign(original_weights[i])
            
            # Update wake for next iteration
            self.generator_wake = [tf.identity(w) for w in model.trainable_weights]
            
            return result
        return wrapper
    return decorator