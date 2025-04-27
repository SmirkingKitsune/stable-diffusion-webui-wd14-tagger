""" This module contains the ui for the tagger tab. """
from typing import Dict, Tuple, List, Optional
import gradio as gr
import re
import json
from pathlib import Path
from PIL import Image
from packaging import version

try:
    from tensorflow import __version__ as tf_version
except ImportError:
    tf_version = '0.0.0'

from html import escape as html_esc

from modules import ui  # pylint: disable=import-error
from modules import generation_parameters_copypaste as parameters_copypaste  # pylint: disable=import-error # noqa

try:
    from modules.call_queue import wrap_gradio_gpu_call
except ImportError:
    from webui import wrap_gradio_gpu_call  # pylint: disable=import-error
from tagger import utils  # pylint: disable=import-error
from tagger import settings
from tagger.interrogator import Interrogator as It  # pylint: disable=E0401
from tagger.uiset import IOData, QData  # pylint: disable=import-error

TAG_INPUTS = ["add", "keep", "exclude", "search", "replace"]
COMMON_OUTPUT = Tuple[
    Optional[str],               # tags as string
    Optional[str],               # html tags as string
    Optional[str],               # discarded tags as string
    Optional[Dict[str, float]],  # rating confidences
    Optional[Dict[str, float]],  # tag confidences
    Optional[Dict[str, float]],  # excluded tag confidences
    str,               # error message
]

class GalleryState:
    """Tracks the state of the gallery across operations."""
    last_update_source: str = None  # 'interrogation' or 'input_glob'
    needs_update: bool = False

def unload_interrogators() -> Tuple[str]:
    unloaded_models = 0
    remaining_models = ''

    for i in utils.interrogators.values():
        if i.unload():
            unloaded_models = unloaded_models + 1
        elif i.model is not None:
            if remaining_models == '':
                remaining_models = f', remaining models:<ul><li>{i.name}</li>'
            else:
                remaining_models = remaining_models + f'<li>{i.name}</li>'
    if remaining_models != '':
        remaining_models = remaining_models + "Some tensorflow models could "\
                           "not be unloaded, a known issue."
    QData.clear(1)

    return (f'{unloaded_models} model(s) unloaded{remaining_models}',)


def on_interrogate(input_glob: str, output_dir: str, name: str, filt: str, *args) -> COMMON_OUTPUT:
    """Handler for interrogation."""
    # Update gallery state
    GalleryState.last_update_source = 'interrogation'
    GalleryState.needs_update = True
    
    # input glob should always be rechecked for new files
    IOData.update_input_glob(input_glob)
    if output_dir != It.input["output_dir"]:
        IOData.update_output_dir(output_dir)
        It.input["output_dir"] = output_dir

    if len(IOData.err) > 0:
        return (None,) * 6 + (IOData.error_msg(),)

    for i, val in enumerate(args):
        part = TAG_INPUTS[i]
        if val != It.input[part]:
            getattr(QData, "update_" + part)(val)
            It.input[part] = val

    interrogator: It = next((i for i in utils.interrogators.values() if
                             i.name == name), None)
    if interrogator is None:
        return (None,) * 6 + (f"'{name}': invalid interrogator",)

    interrogator.batch_interrogate()
    return search_filter(filt)


def on_gallery(gallery) -> List[str]:
    """Handler for gallery tab selection."""
    print(f"Gallery tab selected, source is {GalleryState.last_update_source}")
    
    if GalleryState.last_update_source == "interrogation":
        # Get image paths
        if QData.json_db and QData.json_db.exists():
            image_paths = get_image_paths()
            print(f"Found {len(image_paths)} images for gallery")
            return image_paths
    if GalleryState.last_update_source == "input_glob":
        return gallery

    print("No db.json found")
    return ""
    
def on_interrogate_image(*args) -> COMMON_OUTPUT:
    # hack brcause image interrogaion occurs twice
    It.odd_increment = It.odd_increment + 1
    if It.odd_increment & 1 == 1:
        return (None,) * 6 + ('',)
    return on_interrogate_image_submit(*args)


def on_interrogate_image_submit(
    image: Image, name: str, filt: str, *args
) -> COMMON_OUTPUT:
    for i, val in enumerate(args):
        part = TAG_INPUTS[i]
        if val != It.input[part]:
            getattr(QData, "update_" + part)(val)
            It.input[part] = val

    if image is None:
        return (None,) * 6 + ('No image selected',)
    interrogator: It = next((i for i in utils.interrogators.values() if
                             i.name == name), None)
    if interrogator is None:
        return (None,) * 6 + (f"'{name}': invalid interrogator",)

    interrogator.interrogate_image(image)
    return search_filter(filt)


def move_selection_to_input(
    filt: str, field: str
) -> Tuple[Optional[str], Optional[str], str]:
    """ moves the selected to the input field """
    if It.output is None:
        return (None, None, '')
    tags = It.output[1]
    got = It.input[field]
    existing = set(got.split(', '))
    if filt:
        re_part = re.compile('(' + re.sub(', ?', '|', filt) + ')')
        tags = {k: v for k, v in tags.items() if re_part.search(k) and
                k not in existing}
        print("Tags remaining: ", tags)

    if len(tags) == 0:
        return ('', None, '')

    if got != '':
        got = got + ', '

    (data, info) = It.set(field)(got + ', '.join(tags.keys()))
    return ('', data, info)


def move_selection_to_keep(
    tag_search_filter: str
) -> Tuple[Optional[str], Optional[str], str]:
    return move_selection_to_input(tag_search_filter, "keep")


def move_selection_to_exclude(
    tag_search_filter: str
) -> Tuple[Optional[str], Optional[str], str]:
    return move_selection_to_input(tag_search_filter, "exclude")


def search_filter(filt: str) -> COMMON_OUTPUT:
    """ filters the tags and lost tags for the search field """
    ratings, tags, lost, info = It.output
    if ratings is None:
        return (None,) * 6 + (info,)
    if filt:
        re_part = re.compile('(' + re.sub(', ?', '|', filt) + ')')
        tags = {k: v for k, v in tags.items() if re_part.search(k)}
        lost = {k: v for k, v in lost.items() if re_part.search(k)}

    h_tags = ', '.join(f'<a href="javascript:tag_clicked(\'{html_esc(k)}\','
                       f'true)">{k}</a>' for k in tags.keys())
    h_lost = ', '.join(f'<a href="javascript:tag_clicked(\'{html_esc(k)}\','
                       f'false)">{k}</a>' for k in lost.keys())

    return (', '.join(tags.keys()), h_tags, h_lost, ratings, tags, lost, info)

def update_rating_tags_tab(rating_confidences, tag_confidences, discarded_tags):
    """Updates the Ratings and included tags tab with data from QData."""
    if not QData.weighed:
        return
        
    # Get ratings and tags from QData.weighed
    ratings = QData.weighed[0]  # ratings are in weighed[0]
    tags = QData.weighed[1]     # tags are in weighed[1]
    
    # Update ratings display - calculate average weight per rating
    rating_averages = {}
    for rating, weights in ratings.items():
        if weights:  # Only process if we have weights
            rating_averages[rating] = sum(weight - int(weight) for weight in weights) / len(weights) * 100

    rating_items = sorted(rating_averages.items(), key=lambda x: -x[1])
    ratings_text = "\n".join([f"{k}: {v:.2f}%" for k, v in rating_items])
    rating_confidences.update(value=ratings_text)
    
    # Update tags display - calculate average weight per tag
    tag_averages = {}
    for tag, weights in tags.items():
        if weights:  # Only process if we have weights
            tag_averages[tag] = sum(weight - int(weight) for weight in weights) / len(weights) * 100
    
    # Sort tags by their average weight
    tag_items = sorted(tag_averages.items(), key=lambda x: -x[1])
    
    # Take top 100 tags (or adjust this number as needed. Note: higher numbers might hinder performance)
    top_tags = tag_items[:100000]
    tags_text = "\n".join([f"{k}: {v:.2f}%" for k, v in top_tags])
    tag_confidences.update(value=tags_text)

def get_image_paths() -> List[str]:
    """Gets all image paths from IOData that have associated tags in QData."""
    image_paths = []
    
    # Create a lookup dictionary for faster querying
    query_lookup = {}
    if QData.json_db and QData.json_db.exists():
        try:
            with open(QData.json_db, 'r') as f:
                data = json.load(f)
                query_lookup = {query_data[0]: query_data[1] 
                              for query_data in data.get("query", {}).values()}
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Error reading db.json: {e}")
            return image_paths
    
    # Process paths more efficiently
    for path_info in IOData.paths:
        image_path = str(path_info[0].absolute())
        if image_path in query_lookup:
            image_paths.append(image_path)
            
    return image_paths

def on_input_glob_change(glob_path: str, rating_confidences, tag_confidences, discarded_tags, gallery, info):
    """Handler for input glob changes."""
    error_msg = ""
    
    # First update the input glob in IOData
    IOData.update_input_glob(glob_path)
    
    # Update gallery state
    GalleryState.last_update_source = 'input_glob'
    GalleryState.needs_update = True
    
    # If db.json was loaded, update the UI components
    if QData.json_db and QData.json_db.exists():
        try:
            # Calculate rating confidences
            rating_dict = {}
            for rating, weights in QData.weighed[0].items():
                if weights:
                    avg_weight = sum(weight - int(weight) for weight in weights) / len(weights)
                    rating_dict[rating] = avg_weight
            
            # Calculate tag confidences and apply corrections
            tag_dict = {}
            excluded_dict = {}
            for tag, weights in QData.weighed[1].items():
                if weights:
                    avg_weight = sum(weight - int(weight) for weight in weights) / len(weights)
                    # Apply tag correction before checking exclusion
                    corrected_tag = QData.correct_tag(tag)
                    
                    # Check if tag should be excluded based on QData criteria
                    if QData.is_excluded(corrected_tag) or avg_weight < QData.threshold:
                        excluded_dict[corrected_tag] = avg_weight
                    else:
                        tag_dict[corrected_tag] = avg_weight

            # Sort tags by confidence
            tag_items = sorted(tag_dict.items(), key=lambda x: -x[1])
            excluded_items = sorted(excluded_dict.items(), key=lambda x: -x[1])

            # Create clickable HTML tags
            html_tags = []
            for tag, conf in tag_items[:100000]:  # Limit to top 100 tags (or adjust this number as needed. Note: higher numbers might hinder performance)
                html_tag = f'<a href="javascript:tag_clicked(\'{html_esc(tag)}\',true)">{tag}</a>'
                html_tags.append(html_tag)
            
            html_excluded = []
            for tag, conf in excluded_items[:100000]:  # Limit to top 100 tags (or adjust this number as needed. Note: higher numbers might hinder performance)
                html_tag = f'<a href="javascript:tag_clicked(\'{html_esc(tag)}\',false)">{tag}</a>'
                html_excluded.append(html_tag)

            # Join tags with commas
            tags_text = ', '.join(tag for tag, _ in tag_items[:100000]) # Limit to top 100 tags (or adjust this number as needed. Note: higher numbers might hinder performance)
            html_tags_text = ', '.join(html_tags)
            html_excluded_text = ', '.join(html_excluded)
            
            # Get all images with tags
            gallery_images = get_image_paths()
            print(f"Found {len(gallery_images)} images for gallery")
            
            if gallery_images:
                # Update gallery with absolute paths
                gallery.update(value=gallery_images)
                
                return (
                    glob_path,                # input_glob
                    tags_text,                # tags (plain text)
                    html_tags_text,           # html_tags (clickable)
                    html_excluded_text,       # discarded_tags (clickable)
                    rating_dict,              # rating_confidences
                    tag_dict,                 # tag_confidences
                    excluded_dict,            # excluded_tag_confidences
                    gallery_images,           # gallery images
                    f"Found {len(gallery_images)} images for curation"
                )
            else:
                print("No tagged images found")
                return (
                    glob_path, "", "", "", {}, {}, {},
                    [],  # empty gallery
                    "No tagged images found"
                )
            
        except Exception as e:
            error_msg = f"Error updating UI: {str(e)}"
            print(f"Error in input_glob change handler: {error_msg}")
            import traceback
            traceback.print_exc()
    
    return glob_path, "", "", "", {}, {}, {}, [], error_msg or "No data found"

def update_file_tags(file_path: str, selected_tags: List[str]) -> None:
    """Updates the tags file for a given image with the selected tags."""
    txt_path = Path(file_path).with_suffix('.txt')
    if txt_path.exists():
        tag_string = ', '.join(tag.strip() for tag in selected_tags)
        txt_path.write_text(tag_string, encoding='utf-8')
        print(f"Updated {txt_path} with {len(selected_tags)} tags")

def get_file_tags(file_path: str) -> List[str]:
    """Gets the existing tags from a file's associated txt file."""
    txt_path = Path(file_path).with_suffix('.txt')
    if txt_path.exists():
        content = txt_path.read_text(encoding='utf-8')
        return [tag.strip() for tag in content.split(',') if tag.strip()]
    return []

def parse_weight(weight_str: float) -> Tuple[int, float]:
    """Parses a weight value into image_id and actual weight percentage."""
    image_id = int(weight_str)
    weight = weight_str - image_id
    return image_id, weight * 100  # Convert to percentage

def get_image_tags(image_id: int) -> Dict[str, float]:
    """Gets all tags and their weights for a specific image from db.json."""
    if not QData.json_db or not QData.json_db.exists():
        return {}
    
    try:
        data = json.loads(QData.json_db.read_text())
        tags_data = data.get("tag", {})
        image_tags = {}
        
        for tag, weights in tags_data.items():
            for weight in weights:
                parsed_id, parsed_weight = parse_weight(weight)
                if parsed_id == image_id:
                    corrected_tag = QData.correct_tag(tag)
                    image_tags[corrected_tag] = parsed_weight
        
        return image_tags
        
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error reading db.json: {e}")
        return {}

def get_image_id_from_path(file_path: str) -> Optional[int]:
    """Gets the image ID from the query section of db.json using cached data."""
    if not hasattr(get_image_id_from_path, 'query_cache'):
        get_image_id_from_path.query_cache = {}
        
        if QData.json_db and QData.json_db.exists():
            try:
                with open(QData.json_db, 'r') as f:
                    data = json.load(f)
                    get_image_id_from_path.query_cache = {
                        query_data[0]: query_data[1] 
                        for query_data in data.get("query", {}).values()
                    }
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Error initializing query cache: {e}")
                return None
    
    absolute_path = str(Path(file_path).absolute())
    return get_image_id_from_path.query_cache.get(absolute_path)

def get_sorted_tags(file_path: str) -> List[str]:
    """Gets all tags for an image, sorted by weight and including text file tags."""
    # First get tags from the text file
    file_tags = get_file_tags(file_path)
    file_tags_set = set(file_tags)
    
    # Get tags from db.json
    image_id = get_image_id_from_path(file_path)
    db_tags = []
    if image_id is not None:
        image_tags = get_image_tags(image_id)
        # Sort db tags by weight
        db_tags = [
            tag for tag, _ in sorted(
                image_tags.items(),
                key=lambda x: (-x[1], x[0])  # Sort by weight desc, then tag name asc
            )
        ]
    
    # Combine tags:
    # 1. Start with tags from text file in their original order
    result = file_tags[:]
    # 2. Add tags from db.json that aren't in the text file
    result.extend(tag for tag in db_tags if tag not in file_tags_set)
    
    return result

def get_image_ratings(image_id: int) -> Dict[str, float]:
    """ Gets rating confidences for a specific image from db.json. """
    if not QData.json_db or not QData.json_db.exists():
        return {}
    
    try:
        data = json.loads(QData.json_db.read_text())
        ratings_data = data.get("rating", {})
        image_ratings = {}
        
        for rating, weights in ratings_data.items():
            for weight in weights:
                parsed_id, parsed_weight = parse_weight(weight)
                if parsed_id == image_id:
                    # Store the raw confidence value
                    image_ratings[rating] = parsed_weight
        
        return image_ratings
    
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error reading db.json: {e}")
        return {}

def format_ratings(ratings: Dict[str, float]) -> gr.Label.update:
    """Formats rating confidences to match the Tag Generation tab display format."""
    if not ratings:
        return gr.Label.update(value="No ratings available")
    
    # Create formatted ratings dictionary with proper percentage calculations
    formatted_ratings = {}
    for rating, confidence in ratings.items():
        # Remove 'rating:' prefix if present
        label = rating.replace('rating:', '')
        # Store the original confidence value without additional scaling
        formatted_ratings[label] = confidence / 100
    
    # Sort ratings by confidence (highest first)
    sorted_ratings = sorted(formatted_ratings.items(), key=lambda x: -x[1])
    
    # Create the label update dictionary with properly formatted values
    return gr.Label.update(value={
        k: round(v, 6) for k, v in sorted_ratings
    })

def on_gallery_select(evt: gr.SelectData, state: gr.State) -> tuple:
    """Handler for gallery selection event."""
    image_paths = get_image_paths()
    if not image_paths or evt.index >= len(image_paths):
        return (
            gr.CheckboxGroup.update(choices=[], value=[], label="No image selected"),
            gr.Label.update(value={}),
            None
        )
        
    selected_path = image_paths[evt.index]
    file_tags = get_file_tags(selected_path)
    all_tags = get_sorted_tags(selected_path)
    
    # Get and format rating confidences
    image_id = get_image_id_from_path(selected_path)
    ratings = {}
    if image_id is not None:
        ratings = get_image_ratings(image_id)
        
    rating_update = format_ratings(ratings)
    
    print(f"Selected image: {selected_path}")
    print(f"Found {len(all_tags)} tags, {len(file_tags)} selected")
    print(f"Ratings: {ratings}")
    
    return (
        gr.CheckboxGroup.update(
            choices=all_tags,
            value=file_tags,
            label=f"Tags for {Path(selected_path).name}"
        ),
        rating_update,
        evt.index
    )

def on_tags_change(selected_tags: List[str], selected_index: gr.State) -> None:
    """Handler for checkbox group changes that updates the associated text file."""
    if selected_index is None:
        print("No image selected")
        return
        
    image_paths = get_image_paths()
    if not image_paths or selected_index >= len(image_paths):
        print("Invalid image selection")
        return
        
    selected_path = image_paths[selected_index]
    print(f"Updating tags for {selected_path}")
    print(f"Selected tags: {len(selected_tags)}")
    
    # Update the text file with only the selected tags
    update_file_tags(selected_path, selected_tags)

def add_new_tag(new_tag: str, selected_index: gr.State) -> tuple:
    """Adds a new tag to the image's text file and refreshes the tag editor."""
    if selected_index is None:
        return (
            gr.CheckboxGroup.update(choices=[], value=[]),
            gr.Textbox.update(value="")
        )
    
    image_paths = get_image_paths()
    if not image_paths or selected_index >= len(image_paths):
        return (
            gr.CheckboxGroup.update(choices=[], value=[]),
            gr.Textbox.update(value="")
        )
    
    selected_path = image_paths[selected_index]
    
    # Get existing tags
    current_tags = get_file_tags(selected_path)
    
    # Clean and validate the new tag
    new_tag = new_tag.strip()
    if not new_tag or new_tag in current_tags:
        return (
            gr.CheckboxGroup.update(choices=get_sorted_tags(selected_path), value=current_tags),
            gr.Textbox.update(value="")
        )
    
    # Add the new tag at the beginning of the list
    current_tags.insert(0, new_tag)
    
    # Update the text file
    update_file_tags(selected_path, current_tags)
    
    # Get fresh sorted tags
    all_tags = get_sorted_tags(selected_path)
    
    return (
        gr.CheckboxGroup.update(
            choices=all_tags,
            value=current_tags,
            label=f"Tags for {Path(selected_path).name}"
        ),
        gr.Textbox.update(value="")  # Clear the textbox
    )

def create_gallery_ui(tab_gallery):
    """Creates the gallery UI components."""
    with tab_gallery:
        selected_index = gr.State(None)
        
        # Create a container div for consistent height
        with gr.Box(elem_id="gallery-container"):
            # Use Row for main layout
            with gr.Row():
                # Left column with gallery
                with gr.Column(scale=1):
                    gallery = gr.Gallery(
                        label='Gallery',
                        elem_id='gallery',
                        object_fit="contain",
                        height="800px",
                        show_label=False,
                        columns=4
                    )
                    rating_confidences = gr.Label(
                        label="Rating Confidences",
                        value={},
                        elem_id="rating-confidences"
                    )
                
                # Right column with tag editor
                with gr.Column(scale=1):
                    tag_editor = gr.CheckboxGroup(
                        label="Select image to edit tags",
                        choices=[],
                        value=[],
                        interactive=True,
                        container=True,
                        elem_id="tag-editor"
                    )
                    with gr.Row():
                        new_tag_input = gr.Textbox(
                            label="Add New Tag",
                            placeholder="Enter a new tag",
                            show_label=True,
                            container=True
                        )
                        add_tag_btn = gr.Button(
                            value="+",
                            variant="primary",
                            elem_classes="tool"
                        )
        
        gallery.select(
            fn=on_gallery_select,
            inputs=[selected_index],
            outputs=[tag_editor, rating_confidences, selected_index]
        )
        tag_editor.change(
            fn=on_tags_change,
            inputs=[tag_editor, selected_index],
            outputs=[]
        )
        add_tag_btn.click(
            fn=add_new_tag,
            inputs=[new_tag_input, selected_index],
            outputs=[tag_editor, new_tag_input]
        )
        new_tag_input.submit(
            fn=add_new_tag,
            inputs=[new_tag_input, selected_index],
            outputs=[tag_editor, new_tag_input]
        )
        
        return gallery

def on_ui_tabs():
    """ configures the ui on the tagger tab """
    # If checkboxes misbehave you have to adapt the default.json preset
    tag_input = {}

    with gr.Blocks(analytics_enabled=False) as tagger_interface:
        with gr.Tabs():
            with gr.TabItem("Tag Generation"):
                with gr.Row():
                    with gr.Column(variant='panel'):

                        # input components
                        with gr.Tabs():
                            with gr.TabItem(label='Single process'):
                                image = gr.Image(
                                    label='Source',
                                    source='upload',
                                    interactive=True,
                                    type="pil"
                                )
                                image_submit = gr.Button(
                                    value='Interrogate image',
                                    variant='primary'
                                )

                            with gr.TabItem(label='Batch from directory'):
                                input_glob = utils.preset.component(
                                    gr.Textbox,
                                    value='',
                                    label='Input directory - To recurse use ** or */* '
                                          'in your glob; also check the settings tab.',
                                    placeholder='/path/to/images or to/images/**/*'
                                )
                                output_dir = utils.preset.component(
                                    gr.Textbox,
                                    value=It.input["output_dir"],
                                    label='Output directory',
                                    placeholder='Leave blank to save images '
                                                'to the same path.'
                                )

                                batch_submit = gr.Button(
                                    value='Interrogate',
                                    variant='primary'
                                )
                                with gr.Row(variant='compact'):
                                    with gr.Column(variant='panel'):
                                        large_query = utils.preset.component(
                                            gr.Checkbox,
                                            label='huge batch query (TF 2.10, '
                                            'experimental)',
                                            value=False,
                                            interactive=version.parse(tf_version) ==
                                            version.parse('2.10')
                                        )
                                    with gr.Column(variant='panel'):
                                        save_tags = utils.preset.component(
                                            gr.Checkbox,
                                            label='Save to tags files',
                                            value=True
                                        )

                        info = gr.HTML(
                            label='Info',
                            interactive=False,
                            elem_classes=['info']
                        )

                        # interrogator selector
                        with gr.Column():
                            # preset selector
                            with gr.Row(variant='compact'):
                                available_presets = utils.preset.list()
                                selected_preset = gr.Dropdown(
                                    label='Preset',
                                    choices=available_presets,
                                    value=available_presets[0]
                                )

                                save_preset_button = gr.Button(
                                    value=ui.save_style_symbol
                                )

                                ui.create_refresh_button(
                                    selected_preset,
                                    lambda: None,
                                    lambda: {'choices': utils.preset.list()},
                                    'refresh_preset'
                                )

                            with gr.Row(variant='compact'):
                                def refresh():
                                    utils.refresh_interrogators()
                                    return sorted(x.name for x in utils.interrogators
                                                                       .values())
                                interrogator_names = refresh()
                                interrogator = utils.preset.component(
                                    gr.Dropdown,
                                    label='Interrogator',
                                    choices=interrogator_names,
                                    value=(
                                        None
                                        if len(interrogator_names) < 1 else
                                        interrogator_names[-1]
                                    )
                                )

                                ui.create_refresh_button(
                                    interrogator,
                                    lambda: None,
                                    lambda: {'choices': refresh()},
                                    'refresh_interrogator'
                                )

                            unload_all_models = gr.Button(
                                value='Unload all interrogate models'
                            )
                            with gr.Row(variant='compact'):
                                tag_input["add"] = utils.preset.component(
                                    gr.Textbox,
                                    label='Additional tags (comma split)',
                                    elem_id='additional-tags'
                                )
                            with gr.Row(variant='compact'):
                                threshold = utils.preset.component(
                                    gr.Slider,
                                    label='Weight threshold',
                                    minimum=0,
                                    maximum=1,
                                    value=QData.threshold
                                )
                                tag_frac_threshold = utils.preset.component(
                                    gr.Slider,
                                    label='Min tag fraction in batch and '
                                          'interrogations',
                                    minimum=0,
                                    maximum=1,
                                    value=QData.tag_frac_threshold,
                                )
                            with gr.Row(variant='compact'):
                                cumulative = utils.preset.component(
                                    gr.Checkbox,
                                    label='Combine interrogations from memory',
                                    value=False
                                )
                                merge_existing = utils.preset.component(
                                    gr.Checkbox,
                                    label='Merge interrogation with existing tags',
                                    value=False
                                )
                                
                                unload_after = utils.preset.component(
                                    gr.Checkbox,
                                    label='Unload model after running',
                                    value=False
                                )
                            with gr.Row(variant='compact'):
                                tag_input["search"] = utils.preset.component(
                                    gr.Textbox,
                                    label='Search tag, .. ->',
                                    elem_id='search-tags'
                                )
                                tag_input["replace"] = utils.preset.component(
                                    gr.Textbox,
                                    label='-> Replace tag, ..',
                                    elem_id='replace-tags'
                                )
                            with gr.Row(variant='compact'):
                                tag_input["keep"] = utils.preset.component(
                                    gr.Textbox,
                                    label='Keep tag, ..',
                                    elem_id='keep-tags'
                                )
                                tag_input["exclude"] = utils.preset.component(
                                    gr.Textbox,
                                    label='Exclude tag, ..',
                                    elem_id='exclude-tags'
                                )

                    # output components
                    with gr.Column(variant='panel'):
                        with gr.Row(variant='compact'):
                            with gr.Column(variant='compact'):
                                mv_selection_to_keep = gr.Button(
                                    value='Move visible tags to keep tags',
                                    variant='secondary'
                                )
                                mv_selection_to_exclude = gr.Button(
                                    value='Move visible tags to exclude tags',
                                    variant='secondary'
                                )
                            with gr.Column(variant='compact'):
                                tag_search_selection = utils.preset.component(
                                    gr.Textbox,
                                    label='Multi string search: part1, part2.. '
                                          '(Enter key to update)',
                                )
                        with gr.Tabs():
                            with gr.TabItem(label='Ratings and included tags'):
                                # clickable tags to populate excluded tags
                                tags = gr.State(value="")
                                html_tags = gr.HTML(
                                    label='Tags',
                                    elem_id='tags',
                                )

                                with gr.Row():
                                    parameters_copypaste.bind_buttons(
                                        parameters_copypaste.create_buttons(
                                            ["txt2img", "img2img"],
                                        ),
                                        None,
                                        tags
                                    )
                                rating_confidences = gr.Label(
                                    label='Rating confidences',
                                    elem_id='rating-confidences',
                                )
                                tag_confidences = gr.Label(
                                    label='Tag confidences',
                                    elem_id='tag-confidences',
                                )
                            with gr.TabItem(label='Excluded tags'):
                                # clickable tags to populate keep tags
                                discarded_tags = gr.HTML(
                                    label='Tags',
                                    elem_id='tags',
                                )
                                excluded_tag_confidences = gr.Label(
                                    label='Excluded Tag confidences',
                                    elem_id='discard-tag-confidences',
                                )
            tab_gallery = gr.TabItem(label='Tag Curation')
            gallery = create_gallery_ui(tab_gallery)


        # register events
        # Checkboxes
        cumulative.input(fn=It.flip('cumulative'), inputs=[], outputs=[])
        merge_existing.input(
            fn=lambda x: setattr(QData, 'merge_existing', x),
            inputs=[merge_existing],
            outputs=[]
        )
        large_query.input(fn=It.flip('large_query'), inputs=[], outputs=[])
        unload_after.input(fn=It.flip('unload_after'), inputs=[], outputs=[])

        save_tags.input(fn=IOData.flip_save_tags(), inputs=[], outputs=[])

        # Preset and unload buttons
        selected_preset.change(fn=utils.preset.apply, inputs=[selected_preset],
                               outputs=[*utils.preset.components, info])

        save_preset_button.click(fn=utils.preset.save, inputs=[selected_preset,
                                 *utils.preset.components], outputs=[info])

        unload_all_models.click(fn=unload_interrogators, outputs=[info])

        # Sliders
        threshold.input(fn=QData.set("threshold"), inputs=[threshold],
                        outputs=[])
        threshold.release(fn=QData.set("threshold"), inputs=[threshold],
                          outputs=[])

        tag_frac_threshold.input(fn=QData.set("tag_frac_threshold"),
                                 inputs=[tag_frac_threshold], outputs=[])
        tag_frac_threshold.release(fn=QData.set("tag_frac_threshold"),
                                   inputs=[tag_frac_threshold], outputs=[])

        # Input textboxes (blur == lose focus)
        for tag in TAG_INPUTS:
            tag_input[tag].blur(fn=wrap_gradio_gpu_call(It.set(tag)),
                                inputs=[tag_input[tag]],
                                outputs=[tag_input[tag], info])
        
        # Update UI tabs for input_glob change
        input_glob.blur(
            fn=wrap_gradio_gpu_call(
                lambda x: on_input_glob_change(
                    x,
                    rating_confidences,
                    tag_confidences, 
                    discarded_tags,
                    gallery,
                    info
                )
            ),
            inputs=[input_glob],
            outputs=[
                input_glob,
                tags,               # Plain text tags
                html_tags,          # HTML formatted clickable tags
                discarded_tags,     # HTML formatted clickable excluded tags
                rating_confidences, # Rating confidences dict
                tag_confidences,    # Tag confidences dict
                excluded_tag_confidences, # Excluded tag confidences dict
                gallery,            # Gallery component
                info
            ]
        )
        output_dir.blur(fn=wrap_gradio_gpu_call(It.set("output_dir")),
                        inputs=[output_dir], outputs=[output_dir, info])

        tab_gallery.select(
            fn=on_gallery,
            inputs=[gallery],
            outputs=[gallery]
        )

        common_output = [tags, html_tags, discarded_tags, rating_confidences,
                         tag_confidences, excluded_tag_confidences, info]

        # search input textbox
        for fun in [tag_search_selection.change, tag_search_selection.submit]:
            fun(fn=wrap_gradio_gpu_call(search_filter),
                inputs=[tag_search_selection], outputs=common_output)

        # buttons to move tags (right)
        mv_selection_to_keep.click(
            fn=wrap_gradio_gpu_call(move_selection_to_keep),
            inputs=[tag_search_selection],
            outputs=[tag_search_selection, tag_input["keep"], info])

        mv_selection_to_exclude.click(
            fn=wrap_gradio_gpu_call(move_selection_to_exclude),
            inputs=[tag_search_selection],
            outputs=[tag_search_selection, tag_input["exclude"], info])

        common_input = [interrogator, tag_search_selection] + \
                       [tag_input[tag] for tag in TAG_INPUTS]

        # interrogation events
        image_submit.click(fn=wrap_gradio_gpu_call(on_interrogate_image_submit),
             inputs=[image] + common_input, outputs=common_output)

        image.change(fn=wrap_gradio_gpu_call(on_interrogate_image),
             inputs=[image] + common_input, outputs=common_output)

        batch_submit.click(fn=wrap_gradio_gpu_call(on_interrogate),
                           inputs=[input_glob, output_dir] + common_input,
                           outputs=common_output)

    return [(tagger_interface, "Tagger", "tagger")]
