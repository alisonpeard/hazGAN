import wandb
import tkinter as tk
from tkinter import ttk
import requests
from PIL import Image, ImageTk
from io import BytesIO
import pandas as pd
import numpy as np
import argparse


class ImageRater:
    def __init__(self, project_name, sweep_id, image_key="max_percentiles2"):
        """
        Initialize the image rating system.
        
        Args:
            project_name (str): Name of the W&B project
            sweep_id (str): ID of the sweep to analyze
            image_key (str): Specific image key to look for
        """
        self.api = wandb.Api()
        self.project_name = project_name
        # Extract entity and project separately for later use
        self.entity, self.project = project_name.split('/')
        self.sweep_id = sweep_id
        self.image_key = image_key
        self.current_image_idx = 0
        self.ratings = {}
        
        # Initialize the GUI
        self.root = tk.Tk()
        self.root.title(f"Image Rating System - {image_key}")
        self.setup_gui()
        
        # Load sweep runs
        self.load_sweep_data()

    def load_sweep_data(self):
        """Load all runs from the specified sweep"""
        print(f"Loading sweep data... Looking for images with key: {self.image_key}")
        sweep = self.api.sweep(f"{self.project_name}/{self.sweep_id}")
        self.runs = [run for run in sweep.runs]
        self.images = []
        
        print(f"Found {len(self.runs)} runs")
        for run in self.runs:
            print(f"\nProcessing run {run.id}")
            
            # Look for images in the run history
            history = run.history()
            if self.image_key in history:
                print(f"Found {self.image_key} in run {run.id}")
                values = history[self.image_key].dropna().tolist()
                for value in values:
                    if isinstance(value, dict) and 'path' in value:
                        # Get the correct file path, removing any duplicate "media/images"
                        image_path = value['path'].replace('media/images/media/images', 'media/images')
                        if not image_path.startswith('media/images/'):
                            image_path = f"media/images/{image_path}"
                        
                        # Construct the file URL
                        file_url = f"https://api.wandb.ai/files/{self.project_name}/{run.id}/{image_path}"
                        print(f"Constructed URL: {file_url}")
                        
                        self.images.append({
                            'run_id': run.id,
                            'image_url': file_url,
                            'config': run.config
                        })
        
        print(f"\nTotal {self.image_key} images found: {len(self.images)}")
        if self.images:
            print("First image URL:", self.images[0]['image_url'])
        else:
            print(f"No images with key '{self.image_key}' found!")

    def setup_gui(self):
        """Set up the GUI components"""
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image display
        self.image_label = ttk.Label(self.main_frame)
        self.image_label.grid(row=0, column=0, columnspan=5, pady=10)
        
        # Rating buttons
        for i in range(1, 6):
            btn = ttk.Button(self.main_frame, text=str(i),
                           command=lambda x=i: self.rate_image(x))
            btn.grid(row=1, column=i-1, padx=5)
        
        # Navigation buttons
        ttk.Button(self.main_frame, text="Previous",
                  command=self.prev_image).grid(row=2, column=0, columnspan=2, pady=10)
        ttk.Button(self.main_frame, text="Next",
                  command=self.next_image).grid(row=2, column=3, columnspan=2, pady=10)
        
        # Save button
        ttk.Button(self.main_frame, text="Save Ratings",
                  command=self.save_ratings).grid(row=3, column=0, columnspan=5, pady=10)
        
        # Add status label
        self.status_label = ttk.Label(self.main_frame, text="Loading...")
        self.status_label.grid(row=4, column=0, columnspan=5, pady=10)

    def load_and_display_image(self):
        """Load and display the current image"""
        if not self.images:
            self.status_label.config(text=f"No images with key '{self.image_key}' found!")
            return
            
        if self.current_image_idx < len(self.images):
            try:
                image_data = self.images[self.current_image_idx]
                self.status_label.config(text=f"Loading image {self.current_image_idx + 1}/{len(self.images)}")
                print(f"Attempting to load image from: {image_data['image_url']}")
                
                headers = {"Authorization": f"Bearer {wandb.api.api_key}"}
                response = requests.get(image_data['image_url'], headers=headers)
                response.raise_for_status()
                
                img = Image.open(BytesIO(response.content))
                
                # Resize image to fit display
                img.thumbnail((400, 400))
                photo = ImageTk.PhotoImage(img)
                
                self.image_label.configure(image=photo)
                self.image_label.image = photo  # Keep a reference!
                self.status_label.config(text=f"Image {self.current_image_idx + 1}/{len(self.images)}")
            except Exception as e:
                print(f"Error loading image: {str(e)}")
                self.status_label.config(text=f"Error loading image: {str(e)}")

    def rate_image(self, rating):
        """Store rating for current image"""
        if self.current_image_idx < len(self.images):
            image_data = self.images[self.current_image_idx]
            self.ratings[image_data['run_id']] = {
                'rating': rating,
                'config': image_data['config']
            }
            self.next_image()

    def next_image(self):
        """Show next image"""
        if self.current_image_idx < len(self.images) - 1:
            self.current_image_idx += 1
            self.load_and_display_image()

    def prev_image(self):
        """Show previous image"""
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.load_and_display_image()

    def save_ratings(self):
        """Save ratings to W&B"""
        if not self.ratings:
            self.status_label.config(text="No ratings to save!")
            return
            
        try:
            # Convert ratings to DataFrame
            ratings_data = []
            for run_id, data in self.ratings.items():
                row = {'run_id': run_id, 'rating': data['rating']}
                row.update(data['config'])  # Add hyperparameters
                ratings_data.append(row)
                
            df = pd.DataFrame(ratings_data)
            
            # Initialize wandb with just the project name (not entity/project)
            with wandb.init(entity=self.entity, project=self.project, job_type="analysis") as run:
                # Save the full table
                table = wandb.Table(dataframe=df)
                wandb.log({"ratings_table": table})
                
                # Calculate statistics
                avg_rating = df['rating'].mean()
                rating_std = df['rating'].std()
                
                # Log metrics
                metrics = {
                    'average_rating': avg_rating,
                    'rating_std': rating_std
                }
                
                # Create correlation analysis
                for param in data['config'].keys():
                    if param in df.columns and df[param].dtype in [np.float64, np.int64]:
                        correlation = df['rating'].corr(df[param])
                        metrics[f'correlation_{param}'] = correlation
                
                wandb.log(metrics)
            
            self.status_label.config(text="Ratings saved successfully!")
        except Exception as e:
            self.status_label.config(text=f"Error saving ratings: {str(e)}")
            print(f"Detailed error: {str(e)}")


    def run(self):
        """Start the rating system"""
        self.load_and_display_image()
        self.root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rater settings.')
    parser.add_argument('--user', '-u', type=str, default='alison-peard', help='W&B username')
    parser.add_argument('--project', '-p', type=str, default='hazGAN-linux', help='W&B project name')
    parser.add_argument('--sweep', '-s', type=str, default='5z0rfv65', help='Sweep ID')
    parser.add_argument('--key', '-k', type=str, default='max_samples', help='Image key')
    args = parser.parse_args()

    rater = ImageRater(
        project_name=f"{args.user}/{args.project}",  # Make sure to include your username
        sweep_id=args.sweep,
        image_key=args.key
    )
    rater.run()