import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import gradio as gr
from PIL import Image
import time
import math

torch_dtype = torch.float16

mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

if mps_available:
    device = torch.device("mps")
    torch_dtype = torch.float32

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if mps_available else "cpu")

print(f"device: {device}")

if device == "cpu":
    pipeline_text2image = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
else:
    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch_dtype,
        variant="fp16"
    )

pipeline_image2image = AutoPipelineForImage2Image.from_pipe(pipeline_text2image)

if device == "cpu":
    pipeline_text2image = pipeline_text2image.to(device=device)
    pipeline_image2image = pipeline_image2image.to(device=device)
else:
    pipeline_text2image = pipeline_text2image.to(device=device, dtype=torch_dtype)
    pipeline_image2image = pipeline_image2image.to(device=device, dtype=torch_dtype)

# pipeline_text2image.upcast_vae()
# pipeline_image2image.upcast_vae()

async def generate_image(init_image, prompt, strength, steps, seed=123):
    """
    Generates images from input text only or initial image with input text.
    """
    if init_image is not None:
        # init_image = resize_image(init_image)
        init_image = init_image.resize(size=(512, 512))
        generator = torch.manual_seed(seed)
        end_time = time.time()
        
        if int(steps * strength) < 1:
            steps = math.ceil(1 / max(0.10, strength))
            
        out = pipeline_image2image(
            prompt=prompt,
            image=init_image,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=0.0,
            strength=strength,
            width=512,
            height=512,
            output_type="pil"
        )
    else:
        generator = torch.manual_seed(seed)
        end_time = time.time()
        
        out = pipeline_text2image(
            prompt=prompt,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=0.0,
            width=512,
            height=512,
            output_type="pil"
        )
    
    print(f"Pipeline processing time: {time.time() - end_time} seconds")
    
    nsfw_content_detected = (
        out.nsfw_content_detected[0]
        if "nsfw_content_detected" in out
        else False
    )
    
    if nsfw_content_detected:
        gr.Warning("NSFW content detected")
        return Image.new(mode="RGB", size=(512, 512))
    
    return out.images[0]

with gr.Blocks() as demo:
    init_image_state = gr.State()
    gr.Markdown(
        """
        # SDXL Turbo Text-to-Image and Image-to-Image
        [Code](https://github.com/darylalim/sdxl-turbo-tti-iti)
        """
    )
    with gr.Row():
        prompt = gr.Textbox(
            placeholder="Enter a text prompt",
            label="Text prompt",
            scale=5
        )
        btn = gr.Button(
            "Generate",
            scale=1
        )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                sources=["upload", "webcam", "clipboard"],
                label="Initial image",
                type="pil"
            )
            with gr.Accordion("Options", open=False):
                strength = gr.Slider(
                    label="Strength",
                    value=0.5,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.001
                )
                steps = gr.Slider(
                    label="Steps",
                    value=4,
                    minimum=1,
                    maximum=10,
                    step=1
                )
                seed = gr.Slider(
                    randomize=True,
                    minimum=0,
                    maximum=4294967295,
                    label="Seed",
                    step=1
                )
        with gr.Column():
            image = gr.Image(type="filepath")
    
    inputs = [image_input, prompt, strength, steps, seed]
    btn.click(fn=generate_image, inputs=inputs, outputs=image, show_progress=False)
    prompt.change(fn=generate_image, inputs=inputs, outputs=image, show_progress=False)
    steps.change(fn=generate_image, inputs=inputs, outputs=image, show_progress=False)
    seed.change(fn=generate_image, inputs=inputs, outputs=image, show_progress=False)
    strength.change(fn=generate_image, inputs=inputs, outputs=image, show_progress=False)
    image_input.change(
        fn=lambda x: x,
        inputs=image_input,
        outputs=init_image_state,
        show_progress=False,
        queue=False
    )

demo.queue()

demo.launch()