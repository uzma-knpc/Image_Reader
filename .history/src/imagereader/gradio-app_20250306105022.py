import torch
import multiprocessing
import concurrent.futures
from functools import partial

# Set optimal number of CPU threads globally
NUM_CPU_THREADS = multiprocessing.cpu_count()
torch.set_num_threads(NUM_CPU_THREADS)

def process_image(image):
    # Create thread pool for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CPU_THREADS) as executor:
        # Parallel preprocessing tasks
        future_tasks = {
            'preprocess': executor.submit(preprocess_func, image),
            'extract': executor.submit(extract_func, image),
            'analyze': executor.submit(analyze_func, image)
        }
        
        # Gather results
        results = {k: v.result() for k, v in future_tasks.items()}
    
    return results

# Update your Gradio interface to use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_interface():
    # Move models to GPU if available
    model.to(device)
    
    interface = gr.Interface(
        fn=process_image,
        inputs=gr.Image(),
        outputs=[gr.Text(), gr.Text(), gr.Text()],
        title="Optimized Medical Image Analysis",
        cache_examples=True  # Enable caching for better performance
    )
    
    return interface

if __name__ == "__main__":
    # Set number of workers for Gradio
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        enable_queue=True,
        max_threads=NUM_CPU_THREADS  # Optimize thread usage
    ) 