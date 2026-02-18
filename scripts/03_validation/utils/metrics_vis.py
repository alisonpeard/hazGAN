import pandas as pd
import matplotlib.pyplot as plt
import json

def process_stylegan_logs(file_path):
    # Read and parse the jsonl file
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Extract key metrics
    metrics = pd.DataFrame({
        'tick': [d['Progress/tick']['mean'] for d in data],
        'kimg': [d['Progress/kimg']['mean'] for d in data],
        'loss_G': [d['Loss/G/loss']['mean'] for d in data],
        'loss_D': [d['Loss/D/loss']['mean'] for d in data],
        'real_scores': [d['Loss/scores/real']['mean'] for d in data],
        'fake_scores': [d['Loss/scores/fake']['mean'] for d in data],
        'time_hours': [d['Timing/total_hours']['mean'] for d in data]
    })
    
    return metrics

def plot_training_progress(metrics):
    # Set up the plot style
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot losses
    ax1.plot(metrics['time_hours'], metrics['loss_G'], label='Generator Loss')
    ax1.plot(metrics['time_hours'], metrics['loss_D'], label='Discriminator Loss')
    ax1.set_xlabel('Training Time (hours)')
    ax1.set_ylabel('Loss')
    ax1.set_title('Generator and Discriminator Losses')
    ax1.legend()
    ax1.grid(True)
    
    # Plot real and fake scores
    ax2.plot(metrics['time_hours'], metrics['real_scores'], label='Real Scores')
    ax2.plot(metrics['time_hours'], metrics['fake_scores'], label='Fake Scores')
    ax2.set_xlabel('Training Time (hours)')
    ax2.set_ylabel('Score')
    ax2.set_title('Real and Fake Scores')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    return fig

def print_training_summary(metrics):
    print("Training Summary:")
    print("-" * 50)
    print(f"Total training time: {metrics['time_hours'].max():.2f} hours")
    print(f"Images processed: {metrics['kimg'].max() * 1000:.0f}")
    print("\nFinal metrics:")
    print(f"Generator Loss: {metrics['loss_G'].iloc[-1]:.3f}")
    print(f"Discriminator Loss: {metrics['loss_D'].iloc[-1]:.3f}")
    print(f"Real Scores: {metrics['real_scores'].iloc[-1]:.3f}")
    print(f"Fake Scores: {metrics['fake_scores'].iloc[-1]:.3f}")

def analyze_stylegan_training(file_path):
    # Process the data
    metrics = process_stylegan_logs(file_path)
    
    # Print summary
    print_training_summary(metrics)
    
    # Create and show plots
    fig = plot_training_progress(metrics)
    
    return metrics, fig

# Usage example:
if __name__ == "__main__":
    file_path = "stats.jsonl"
    metrics, fig = analyze_stylegan_training(file_path)
    plt.show()
    
    # Save the plots if needed
    # fig.savefig('stylegan_training_progress.png', dpi=300, bbox_inches='tight')