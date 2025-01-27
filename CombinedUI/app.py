import gradio as gr
import os
import tempfile
import logging
import sys
import traceback
import torch
import cv2
import numpy as np
from pose_processing import PoseProcessor
from pose_models import FourDHumanWrapper
from RuleExtration import real_time_debug, combine_video

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pose_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
#load models
FDHuman_wrapper = FourDHumanWrapper()


# Pose processing models:
def verify_output(path):
    """Verify if output video exists"""
    exists = os.path.exists(path)
    logger.info(f"Verifying path {path}: {'exists' if exists else 'does not exist'}")
    return exists

def process_video(video_path, selected_models, noise_filter = 'Original'):
    try:
        output_paths = {}
        video_dir = os.path.dirname(video_path)
        logger.info(f"Processing video in directory: {video_dir}")

        for model in selected_models:
            match model:
                case "mediapipe":
                    processor = PoseProcessor()
                    #use mediapipe without additional methods
                    output_path = f'./output_{noise_filter}.mp4'
                    processor.process_video(video_path, method=noise_filter)
                    output_paths[model] = output_path
                    logger.info(f"Expected output path: {output_path}")
                case "fourdhumans":
                    output_path = f'./output_{model}.mp4'
                    FDHuman_wrapper.process_video(video_path, output_path,noise_filter)
                    output_paths[model] = output_path
                    logger.info(f"Expected output path: {output_path}")
        return output_paths
    except Exception as e:
        logger.error(f"Error processing video with message: {str(e)}")
        logger.error(traceback.format_exc())
        gr.Warning(f"Error processing video: {str(e)}")
        return None

    

def apply_method(video_path, selected_methods):    
    try:
        processor = PoseProcessor()
        output_paths = {}
        video_dir = os.path.dirname(video_path)
        logger.info(f"Processing video in directory: {video_dir}")
        
        # Process each selected method
        for method in selected_methods:
            processing_method = 'original' if method == 'no interpolation' else method
            logger.info(f"Processing method {processing_method}")
            output_path = f'./output_{processing_method}.mp4'
            processor.process_video(video_path, method=processing_method)
            output_paths[method] = output_path
            logger.info(f"Expected output path: {output_path}")
            
            # Verify output was created
            if not verify_output(output_path):
                logger.error(f"Output file not created for method {processing_method}")
                raise Exception(f"Failed to create output for {processing_method}")
        return output_paths
    
    except Exception as e:
        logger.error(f"Error processing video with message: {str(e)}")
        logger.error(traceback.format_exc())
        gr.Warning(f"Error processing video: {str(e)}")
        return None

def create_ui():
    try:
        with gr.Blocks(title="Pose Interpolation Comparison") as app:
            gr.Markdown("# Pose Landmark Interpolation Comparison")
            
            # Define choices as a list of strings
            MODEL_CHOICES = [
                'mediapipe', 'fourdhumans'
            ]
            INTERPOLATION_METHODS = [
                'no interpolation', 'kalman', 'wiener',
                'linear', 'bilinear', 'spline', 'kriging'
            ]
            NOISE_METHODS = [
                'original','chebyshev','bessel','butterworth'
            ]
            # Input controls
            with gr.Row():
                input_video = gr.Video(
                    label="Upload or Record Video",
                    sources=["upload", "webcam"],
                    format="mp4",
                    interactive=True
                )
            
            with gr.Tab("Model comparison"):
                model_columns={}
                model_components={}
                # select model to run on:
                with gr.Row():
                    model_checkbox = gr.CheckboxGroup(
                        choices=MODEL_CHOICES
                    )
                with gr.Row():
                    model_noise_radio = gr.Radio(
                        choices=NOISE_METHODS,
                        label="Select Noise Filter",
                        value='original'
                    )
                # with gr.Row():
                #     gr.Label("Mesh models and noise methods interations are bit wreid right now and often results in bugs. Revisiting the code to fix this issue on Wednesday.")
                with gr.Row():
                    process_btn_model = gr.Button("Process Video", variant="primary")
                    status_model = gr.Textbox(label="Status", interactive=False)
                with gr.Row():
                    for i in range(len(MODEL_CHOICES)):
                        model_columns[MODEL_CHOICES[i]] = gr.Column(visible=True)
                        with model_columns[MODEL_CHOICES[i]]:
                            model_components[MODEL_CHOICES[i]] = gr.Video(
                                label=MODEL_CHOICES[i],
                                interactive=False,
                                width = 400,
                                height=300
                            )


            with gr.Tab("Noise"):
                noise_columns={}
                noise_components={}
                with gr.Row():
                    noise_checkbox = gr.CheckboxGroup(
                        choices=NOISE_METHODS
                    )
                with gr.Row():
                    process_btn_noise = gr.Button("Process Video", variant="primary")
                    status_noise = gr.Textbox(label="Status", interactive=False)
                with gr.Row():
                    for i in range(len(NOISE_METHODS)):
                        noise_columns[NOISE_METHODS[i]] = gr.Column(visible=True)
                        with noise_columns[NOISE_METHODS[i]]:
                            noise_components[NOISE_METHODS[i]] = gr.Video(
                                label=NOISE_METHODS[i],
                                interactive=False,
                                width = 400,
                                height=300
                            )
                    
            with gr.Tab("Interpolation"):
                # Interpolation method selection
                with gr.Row():
                    interpolation_checkbox = gr.CheckboxGroup(
                        choices=INTERPOLATION_METHODS,
                        label="Select Interpolation",
                        value=[]
                    )
                with gr.Row():
                    process_btn_interp = gr.Button("Process Video", variant="primary")
                    status_interpolation = gr.Textbox(label="Status", interactive=False)
                
                        
                # Dictionary to store video components
                video_components_interp = {}
                video_columns_interp = {}
                # # Initialize all possible video components (hidden initially)
                row_count = (len(INTERPOLATION_METHODS) + 3) // 4
                count = 0
                for i in range(row_count):
                    with gr.Row():
                        curr_methods = INTERPOLATION_METHODS[count:count+4]

                        for method in curr_methods:
                            # Create a formatted label
                            if method == 'no interpolation':
                                label = 'No Interpolation'
                            else:
                                label = method.capitalize() + ' Filter' if method in ['kalman', 'butterworth', 'wiener'] else method.capitalize() + ' Interpolation'
                            video_columns_interp[method] = gr.Column(visible=False)
                            with video_columns_interp[method]:
                                video_components_interp[method] = gr.Video(
                                    label=label,
                                    interactive=False,
                                    width = 400,
                                    height=300
                                    # visible=False
                                )
                                count+=1
            with gr.Tab("Rule Extraction"):
                with gr.Row():
                    rule_extraction_btn = gr.Button("Extract Rules", variant="primary")
                    status_rule_extraction = gr.Textbox(label="Status", interactive=False)
                with gr.Row():
                    rule_extraction_output = gr.Video(
                        interactive=False,
                        width = 1800,
                        height= 600
                    )
            
            def layout_videos_interp(selected_methods):
                updates = {}
                # Calculate rows for 4 videos per row
                num_selected = len(selected_methods)
                num_rows = (num_selected + 3) // 4
                logger.info(f"Number of rows {num_rows}, for selected methods: {str(selected_methods)}")

                # # First, hide all videos
                # for component in video_components.values():
                #     updates[component] = gr.Video(visible=False)
                for compoment in video_columns_interp.values():
                    updates[compoment] = gr.Column(visible=False)
                # not_selected = set(INTERPOLATION_METHODS) - set(selected_methods)
                # for curr in not_selected:
                #     video_columns[curr] = gr.Column(visible=False)
                # Show only selected videos
                count = 0
                for i in range(num_rows):
                    # with gr.Row():
                    curr_methods = selected_methods[count:count+4]
                    for method in curr_methods:
                        # video_columns[method] = gr.Column()
                        # with video_columns[method]:
                            # video_components[method] = gr.Video(
                            #     label= method,
                            #     interactive=False
                            # )
                            # logger.info(method)

                        updates[video_columns_interp[method]] = gr.Column(visible=True)

                            # updates[video_components[method]] = gr.Video(visible=True)
                        count+=1
                logger.info(str(count))
                updates[status_interpolation] = gr.Textbox(value=f"Showing {len(selected_methods)} video components")
                return updates

            def process_videos_interp(video_path, selected_methods):
                updates = {}
                
                if not selected_methods:
                    for comp in video_components_interp.values():
                        updates[comp] = gr.Video(value=None)
                    updates[status_interpolation] = gr.Textbox(value="Please select interpolation methods")
                    return updates
                
                if video_path is None:
                    for comp in video_components_interp.values():
                        updates[comp] = gr.Video(value=None)
                    updates[status_interpolation] = gr.Textbox(value="Please select a video")
                    return updates
                
                try:
                    # Process the video
                    output_paths = apply_method(video_path, selected_methods)
                    
                    if output_paths is None:
                        for comp in video_components_interp.values():
                            updates[comp] = gr.Video(value=None)
                        updates[status_interpolation] = gr.Textbox(value="Processing failed!")
                        return updates
                    
                    # Update each component with its corresponding output
                    for method, comp in video_components_interp.items():
                        updates[comp] = gr.Video(
                            value=output_paths.get(method),
                            visible=(method in selected_methods)
                        )
                            
                    updates[status_interpolation] = gr.Textbox(value="Processing complete!")
                    return updates
                    
                except Exception as e:
                    logger.error(f"Error in process_videos: {str(e)}")
                    updates[status_interpolation] = gr.Textbox(value=f"Error: {str(e)}")
                    return updates
            
            def process_videos_model(video_path, selected_models, model_noise_radio):
                updates = {}
                if not selected_models:
                    for comp in model_components.values():
                        updates[comp] = gr.Video(value=None)
                    updates[status_model] = gr.Textbox(value="Please select a model")
                    return updates
                if video_path is None:
                    for comp in model_components.values():
                        updates[comp] = gr.Video(value=None)
                    updates[status_model] = gr.Textbox(value="Please select a video")
                    return updates
                try:
                    # Process the video
                    
                    output_paths = {}
                    if model_noise_radio == 'original':
                        for model in selected_models:
                            output_paths = process_video(video_path, selected_models)
                    else:
                        output_paths = process_video(video_path, selected_models,model_noise_radio)
                        
                    # output_paths = process_video(video_path, selected_models) # TEMPORARY
                    if output_paths is None:
                        for comp in model_components.values():
                            updates[comp] = gr.Video(value=None)
                        updates[status_model] = gr.Textbox(value="Processing failed!")
                        return updates
                    
                    # Update each component with its corresponding output
                    for model, comp in model_components.items():
                        updates[comp] = gr.Video(
                            value=output_paths.get(model),
                            visible=(model in selected_models)
                        )

                    updates[status_model] = gr.Textbox(value="Processing complete!")
                    return updates
                except Exception as e:
                    logger.error(f"Error in process_videos: {str(e)}")
                    updates[status_noise] = gr.Textbox(value=f"Error: {str(e)}")
                    return updates


            def process_videos_noise(video_path, selected_methods):
                updates = {}
                
                if not selected_methods:
                    for comp in noise_components.values():
                        updates[comp] = gr.Video(value=None)
                    updates[status_noise] = gr.Textbox(value="Please select Noise Filter methods")
                    return updates
                
                if video_path is None:
                    for comp in noise_components.values():
                        updates[comp] = gr.Video(value=None)
                    updates[status_noise] = gr.Textbox(value="Please select a video")
                    return updates
                
                try:
                    # Process the video
                    output_paths = apply_method(video_path, selected_methods)
                    
                    if output_paths is None:
                        for comp in noise_components.values():
                            updates[comp] = gr.Video(value=None)
                        updates[status_noise] = gr.Textbox(value="Processing failed!")
                        return updates
                    
                    # Update each component with its corresponding output
                    for method, comp in noise_components.items():
                        updates[comp] = gr.Video(
                            value=output_paths.get(method),
                            visible=(method in selected_methods)
                        )
                            
                    updates[status_noise] = gr.Textbox(value="Processing complete!")
                    return updates
                    
                except Exception as e:
                    logger.error(f"Error in process_videos: {str(e)}")
                    updates[status_noise] = gr.Textbox(value=f"Error: {str(e)}")
                    return updates


            def update_video_status(video_path):
                return {
                    status_interpolation: gr.Textbox(
                        value="Video selected. Click 'Process Video' to start processing." 
                        if video_path else "Please select a video first"
                    )
                }
            def update_noise_radio(noise_radio):
                return {
                    status_noise: gr.Textbox(
                        value=f'Noise Filter {noise_radio} selected.' 
                        if noise_radio else "Please select a Noise Filter first"
                    )
                }
            
            def process_videos_ruleExtraction(video_path):
                updates = {}
                if video_path is None:
                    updates[status_rule_extraction] = gr.Textbox(value="Please select a video")
                    updates[rule_extraction_output] = gr.Textbox(value="No video selected")
                    return updates
                try:
                    # Process the video
                    filename = os.path.splitext(os.path.basename(video_path))[0]
                    output_path = f'./{filename}_combined.mp4'
                    filtered_path,features_path = real_time_debug.process_video_direct(video_path)
                    combine_video.combine_videos(video_path,filtered_path,features_path,output_path)
                    updates[status_rule_extraction] = gr.Textbox(value="Processing complete!")
                    updates[rule_extraction_output] = gr.Textbox(value=output_path)
                    return updates
                except Exception as e:
                    logger.error(f"Error in process_videos: {str(e)}")
                    updates[status_rule_extraction] = gr.Textbox(value=f"Error: {str(e)}")
                    updates[rule_extraction_output] = gr.Textbox(value="Error processing video")
                    return updates
            
            # Connect components
            input_video.change(
                fn=update_video_status,
                inputs=[input_video],
                outputs=[status_interpolation]
            )

            # Update video layout when checkbox changes
            interpolation_checkbox.change(
                fn=layout_videos_interp,
                inputs=[interpolation_checkbox],
                # outputs=list(video_components.values()) + [status_interpolation]
                outputs=list(video_columns_interp.values())+[status_interpolation]
            )
            # model_noise_radio.change(
            #     fn = update_noise_radio,
            #     inputs=[model_noise_radio],
            #     outputs=[status_model]
            # )
            # Handle video processing
            process_btn_model.click(
                fn=process_videos_model,
                inputs=[input_video, model_checkbox, model_noise_radio],
                outputs=list(model_components.values()) + [status_model]
            )

            process_btn_noise.click(
                fn=process_videos_noise,
                inputs=[input_video, noise_checkbox],
                outputs=list(noise_components.values()) + [status_noise]
            )

            process_btn_interp.click(
                fn=process_videos_interp,
                inputs=[input_video, interpolation_checkbox],
                outputs=list(video_components_interp.values()) + [status_interpolation]
            )
            
            rule_extraction_btn.click(
                fn=process_videos_ruleExtraction,
                inputs=[input_video],
                outputs=[status_rule_extraction, rule_extraction_output]
            )
            
        return app

    except Exception as e:
        logger.error(f"Error creating UI: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting application")
        app = create_ui()
        app.launch(share=False)
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        raise