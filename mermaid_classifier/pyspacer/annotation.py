"""
Get/generate and show annotations for a specified image.
"""
from collections import defaultdict
import colorsys
import csv
import enum
from io import BytesIO, StringIO
from operator import itemgetter
import os
from pathlib import Path
import re
from urllib.parse import urlparse
import urllib.request

from bs4 import BeautifulSoup
import matplotlib as mpl
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from spacer.extractors import EfficientNetExtractor
from spacer.messages import ClassifyImageMsg, DataLocation
from spacer.storage import load_image, storage_factory
from spacer.tasks import classify_image

from mermaid_classifier.common.benthic_attributes import (
    BenthicAttributeLibrary, GrowthFormLibrary)
from mermaid_classifier.pyspacer.settings import settings
from mermaid_classifier.pyspacer.utils import mlflow_connect


# A model ID is m- followed by 32 hex digits, but we'll assume 30-32
# hex digits is meant to be an MLflow run ID.
# That way, if the user accidentally copies only 31 digits and
# pastes it into this script, it'll try to find the run ID instead of
# trying it as a filesystem path.
MLFLOW_MODEL_ID_REGEX = re.compile(r'm-[a-f0-9]{30,32}')


def mlflow_model_id_to_pkl_uri(model_id: str) -> str:

    model_filename = 'model.pkl'
    mlflow_connect()

    logged_model = mlflow.get_logged_model(model_id)
    return logged_model.artifact_location + '/' + model_filename


class MarkerShape(enum.Enum):
    BOX = 'box'
    CROSS = 'cross'


class AnnotationRun:

    def __init__(
        self,
        image: str,
        points_csv: str,
        classifier: str = None,
        weights: str = None,
        labelset_csv: str = None,
        num_predictions_to_save: int = 0,
        coralnet_cache_dir: str = None,
        marker_shape: str = MarkerShape.BOX.value,
        plot_title: str = None,
    ):
        """
        image

        Image file to overlay the points on. Can be a local file path,
        an S3 URI, or a CoralNet public image ID.

        points_csv

        Local path to CSV file with points. Accepted column names:
        row, column, label, score. label and score are optional; scores
        should be between 0.0 and 1.0.

        classifier

        Classifier file location.
        Alternatively can come from environment variable
        CLASSIFIER_LOCATION (but this command line arg has higher
        priority).
        Can be a local filepath, an S3 URI, or an MLflow run ID (in which
        case the MLflow tracking server must be running).
        If a classifier is provided, it will be used to classify the
        given image-points, thus generating labels for each point.
        Else, the labels for each point must come from points_csv.

        weights

        EfficientNet extractor weights file location.
        Alternatively can come from environment variable
        WEIGHTS_LOCATION (but this command line arg has higher
        priority).
        Can be a local path, or an S3 URI.
        Must be provided if classifier_location is provided.

        labelset_csv

        Local path to a CSV file which maps label IDs to the names
        you want to display on the visualizer.
        Accepted column names: id, name.
        If not specified, the MERMAID API is used to get the names.

        num_predictions_to_save

        If this is not 0 (the default), classifier predictions are saved
        back to the points_csv. Can be useful when comparing a
        classifier's predictions with manual annotations (or with
        another classifier).
        The value of this option determines how many predictions are saved
        per point. Top predictions use the label and score columns,
        #2 predictions use the label2 and score2 columns, and so on.

        coralnet_cache_dir

        Local path to a directory to be used for caching images downloaded
        from CoralNet. Can be useful when doing multiple runs
        on the same CoralNet image.

        marker_shape

        Shape to use for the point markers which are drawn on the image to
        indicate the point locations.
        See the MarkerShape Enum for the available choices.

        plot_title

        Title at the top of the plot that visualizes the annotations.
        If not given, a title will be generated automatically based on
        inputs.
        """
        self.points_csv_path = points_csv
        self.num_predictions_to_save = num_predictions_to_save
        self.coralnet_cache_dir = coralnet_cache_dir
        self.marker_shape = marker_shape

        weights_location = (
            weights or settings.weights_location)

        annotations: dict[tuple[int, int], list] = defaultdict(list)
        scores: dict[tuple[int, int], list] = defaultdict(list)

        # Read in CSV points.
        # Labels and scores may or may not be present.
        with open(self.points_csv_path) as points_csv:
            reader = csv.DictReader(points_csv)
            for csv_row in reader:
                row = int(csv_row['row'])
                column = int(csv_row['column'])
                # Optional cells could not be in the dict or they could be ''.
                # Default label is 'None'. Default score is nothing (not added
                # to dict).
                annotations[(row, column)].append(csv_row.get('label', 'None'))
                if csv_row.get('score'):
                    scores[(row, column)].append(float(csv_row['score']))

        try:
            image_id = int(image)
        except ValueError:
            # Non-numeric arg; interpret as file path/URI
            self.image_loc = self.parse_location_str(image)
            auto_plot_title = image
        else:
            # Numeric arg; interpret as CoralNet image ID
            self.image_loc = self.get_coralnet_image(image_id)
            auto_plot_title = f"CoralNet image {image_id}"

        classifier_loc = self.parse_location_str(
            classifier,
            is_classifier=True,
        )
        if classifier_loc:
            auto_plot_title += f"\nClassified by: {classifier}"
        else:
            auto_plot_title += f"\nAnnotations from CSV file"

        self.plot_title = plot_title or auto_plot_title

        weights_loc = self.parse_location_str(weights_location)

        if classifier_loc:

            # Classify with pyspacer

            message = ClassifyImageMsg(
                job_token=f'{image}_classify',
                image_loc=self.image_loc,
                extractor=EfficientNetExtractor(
                    data_locations=dict(weights=weights_loc),
                ),
                rowcols=[rowcol for rowcol in annotations.keys()],
                classifier_loc=classifier_loc,
            )

            print("Running classify_image()...")
            return_message = classify_image(message)
            print("Finished classifying")

            predictions_per_point = max(num_predictions_to_save, 1)

            # Get top label(s) and score(s) for each point
            labels = return_message.classes
            for i, (row, column, msg_scores) in enumerate(
                return_message.scores
            ):
                top_predictions = sorted(
                    zip(labels, msg_scores), key=itemgetter(1), reverse=True)
                annotations[(row, column)] = [
                    label for label, score
                    in top_predictions[:predictions_per_point]]
                scores[(row, column)] = [
                    score for label, score
                    in top_predictions[:predictions_per_point]]

        else:

            print(
                "No classifier specified, so points_csv must specify"
                " all annotations.")

        # Number 1 prediction for each point.
        self.top_annotations = dict(
            (rowcol, labels[0]) for rowcol, labels in annotations.items())
        self.top_scores = dict(
            (rowcol, score_vals[0]) for rowcol, score_vals in scores.items())

        # All points must have labels specified by now - either through the
        # points CSV, or using the classifier (the latter takes precedence if
        # specified).
        for rowcol, top_label in self.top_annotations.items():
            if top_label == 'None':
                raise ValueError(f"Point {rowcol} doesn't have a label.")

        if self.num_predictions_to_save > 0 and classifier:
            # Saving predictions back to CSV is enabled, and is applicable
            # since a classifier is being used.
            self.write_predictions(annotations, scores)

        unique_top_labels = set(
            label for label in self.top_annotations.values())

        if labelset_csv:
            # Read in the label IDs to names mapping.
            with open(labelset_csv) as csv_f:
                reader = csv.DictReader(csv_f)
                self.label_ids_to_names = {
                    csv_row['id']: csv_row['name']
                    for csv_row in reader
                    if csv_row['id'] in unique_top_labels
                }
        else:
            # Use MERMAID's API to get the names.
            ba_library = BenthicAttributeLibrary()
            gf_library = GrowthFormLibrary()
            self.label_ids_to_names = dict()
            for bagf_id in unique_top_labels:
                name = ba_library.bagf_id_to_name(bagf_id, gf_library)
                self.label_ids_to_names[bagf_id] = name

    @staticmethod
    def parse_location_str(
        location: str, is_classifier: bool = False,
    ) -> DataLocation:

        if not location:
            return None

        if is_classifier:
            match = MLFLOW_MODEL_ID_REGEX.fullmatch(location)
            if match:
                # MLflow registered model ID.
                # This'll require the tracking server to be running.
                model_id = location
                location = mlflow_model_id_to_pkl_uri(model_id)

        # Now we may or may not have an MLflow-proxied URI
        # beginning with `mlflow-artifacts:/`.
        # If we do, un-proxy it.
        # MLflow probably has 5 ways of un-proxying such artifact URIs, but
        # none have worked for us for some reason, so we un-proxy it manually.
        uri = urlparse(location)
        if uri.scheme == 'mlflow-artifacts':
            # So far we only handle the case where artifacts are stored in
            # the local filesystem cwd.
            # If other cases come up in practice, add to this code to handle
            # those cases.
            location = 'mlartifacts' + uri.path

        try:
            # S3 URI
            # Example:
            # s3://my-bucket/my-folder/model.pkl
            uri = urlparse(location)
            if uri.scheme == 's3':
                return DataLocation(
                    's3',
                    bucket_name=uri.netloc,
                    # url.path probably begins with a slash, which isn't
                    # what we want when ultimately passing this to boto's
                    # Object().
                    key=uri.path.strip('/'),
                )
        except ValueError:
            pass

        # Default to assuming a filesystem path
        return DataLocation('filesystem', key=location)

    def get_coralnet_image(self, image_id: int) -> DataLocation:
        """
        Return a DataLocation whose file contents are the CoralNet image of
        ID `image_id`.
        """
        if self.coralnet_cache_dir:
            # We don't know the file suffix, so look for any file that has the
            # expected filename without the suffix.
            for filename in os.listdir(self.coralnet_cache_dir):
                if Path(filename).stem == f'i{image_id}':
                    print("CoralNet image found in the cache dir")
                    image_path = Path(self.coralnet_cache_dir, filename)
                    return DataLocation('filesystem', str(image_path))

        image_view_url = f'https://coralnet.ucsd.edu/image/{image_id}/view/'
        image_view_response = urllib.request.urlopen(image_view_url)
        response_soup = BeautifulSoup(
            image_view_response.read(), 'html.parser')

        original_img_elements = response_soup.select(
            'div#original_image_container > img')
        if not original_img_elements:
            # This only happens on a private source. If the image is
            # nonexistent, then the response is 404, causing urlopen() to
            # fail with an HTTPError.
            raise ValueError(
                f"CoralNet image {image_id}: couldn't find image on the"
                f" image-view page. Maybe it's in a private source.")
        image_url = original_img_elements[0].attrs.get('src')
        file_suffix = Path(urlparse(image_url).path).suffix

        print("Downloading CoralNet image...")
        download_response = urllib.request.urlopen(image_url)
        print("Finished download")

        if self.coralnet_cache_dir:
            image_path = Path(
                self.coralnet_cache_dir, f'i{image_id}{file_suffix}')
            with open(image_path, 'wb') as image_file:
                image_file.write(download_response.read())
            return DataLocation('filesystem', str(image_path))
        else:
            # No cache
            image_loc = DataLocation('memory', 'image')
            memory_storage = storage_factory('memory')
            memory_storage.store(
                image_loc.key, BytesIO(download_response.read()))
        return image_loc

    @staticmethod
    def prediction_column_names(prediction_num):
        if prediction_num == 1:
            return 'label', 'score'
        else:
            return f'label{prediction_num}', f'score{prediction_num}'

    def write_predictions(self, annotations, scores):

        writer_stream = StringIO()

        # Iterate over a reader of the points-csv file while writing
        # into a memory stream.
        with open(self.points_csv_path) as points_csv:
            reader = csv.DictReader(points_csv)

            # Output CSV will have these columns first.
            writer_fieldnames = ['row', 'column']
            for num in range(1, self.num_predictions_to_save + 1):
                writer_fieldnames += self.prediction_column_names(num)
            # Then, any additional columns that the original CSV happens
            # to have.
            writer_fieldnames += [
                name for name in reader.fieldnames
                if name not in writer_fieldnames]
            writer = csv.DictWriter(writer_stream, writer_fieldnames)
            writer.writeheader()

            # For each original CSV row, add a row to the new CSV
            # content. Note how this keeps the rows in the same order.
            for old_csv_row in reader:
                new_csv_row = old_csv_row.copy()

                row = int(old_csv_row['row'])
                column = int(old_csv_row['column'])

                for index in range(self.num_predictions_to_save):
                    prediction_num = index + 1
                    annotation = annotations[(row, column)][index]
                    score_str = format(scores[(row, column)][index], '.4f')

                    label_col_name, score_col_name = \
                        self.prediction_column_names(prediction_num)
                    new_csv_row[label_col_name] = annotation
                    new_csv_row[score_col_name] = score_str

                writer.writerow(new_csv_row)

        # Write that memory stream back to the points-csv file.
        with open(
            self.points_csv_path, 'w', newline='', encoding='utf-8'
        ) as points_csv:
            points_csv.write(writer_stream.getvalue())

    @property
    def marker_shape_choices(self):
        return [enum_member.value for enum_member in MarkerShape]

    @property
    def marker_shape_choices_str(self):
        return ', '.join(self.marker_shape_choices)

    @staticmethod
    def scale_lightness(rgba, scale_l):
        """
        https://stackoverflow.com/a/60562502
        """
        r, g, b, a = rgba
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        new_rgb = colorsys.hls_to_rgb(h, min(1.0, l * scale_l), s=s)
        return [*new_rgb, a]

    def show(self):

        shadow_color = 'black'

        unique_top_labels = list(self.label_ids_to_names.keys())
        # Use the tab10 color set as a base, but make the colors lighter/darker
        # to go better with dark/light shadows and outlines.
        color_set = mpl.cm.tab10(range(len(unique_top_labels)))
        if shadow_color == 'black':
            scale_l = 4/3
        else:
            # white
            scale_l = 3/4
        color_set = [self.scale_lightness(c, scale_l) for c in color_set]
        # Map the labelset to colors in the color set.
        colors = dict(zip(
            unique_top_labels,
            color_set,
        ))

        fig = plt.gcf()
        ax = fig.add_subplot(111)
        ax.set_title(self.plot_title)

        image = load_image(self.image_loc)
        smaller_dim = min([image.width, image.height])
        # Add image to the plot; this seems to also reverse the y axis
        # (which we want) and force the correct aspect ratio.
        image_array = np.asarray(image)
        plt.imshow(image_array)

        shadow_scatters = []
        main_scatters = []

        # Add point annotations as scatter plots, with each label getting its
        # own color. By doing one scatter-plot action per label, we facilitate
        # creation of the legend.
        for label in unique_top_labels:
            label_rowcols = [
                rowcol for rowcol, top_label in self.top_annotations.items()
                if top_label == label
            ]

            # Columns
            xs = [rowcol[1] for rowcol in label_rowcols]
            # Rows
            ys = [rowcol[0] for rowcol in label_rowcols]

            scatter_common_props = dict()

            # https://matplotlib.org/stable/api/markers_api.html
            match self.marker_shape:

                case MarkerShape.BOX.value:
                    # Square shape.
                    scatter_common_props['marker'] = 's'
                    # Patch-sized box, like in MERMAID
                    # TODO: This isn't at all working; zooming in/out doesn't
                    #  scale the marker
                    marker_size = EfficientNetExtractor.CROP_SIZE
                    # Distance offset of the score number from the marker,
                    # if confidence scores are being shown
                    score_offset = marker_size / 40.0
                    # Outline only, not a filled box
                    scatter_common_props['facecolor'] = 'none'

                case MarkerShape.CROSS.value:
                    scatter_common_props['marker'] = '+'
                    marker_size = smaller_dim / 15
                    score_offset = marker_size / 50.0
                    # + is already an unfilled shape, so don't need to
                    # clear the facecolor.

                case _:
                    raise ValueError(
                        f"Marker shape choice should be one of:"
                        f" {self.marker_shape_choices_str}")

            # 'Outline' of the marker. This is just a slightly bigger black
            # marker drawn before the main marker. This additional color lets
            # points stand out from the background more consistently.
            #
            # TODO: Can this be merged into the legend?
            #  See "legend entries with more than one legend key"
            #  https://matplotlib.org/stable/gallery/text_labels_and_annotations/legend_demo.html
            #  We seem to be halfway there, but need to fix the positioning
            #  of the two markers relative to each other in the legend.
            shadow_scatters.append(ax.scatter(
                xs,
                ys,
                s=marker_size*1.15,
                linewidths=3,
                # Marker color
                color=shadow_color,
                **scatter_common_props
            ))

            # Main marker. Append to the array that will go into the legend.
            main_scatters.append(ax.scatter(
                xs,
                ys,
                s=marker_size,
                linewidths=2,
                # Marker color and shape
                color=colors[label],
                **scatter_common_props
            ))

            # Write scores next to markers
            for rowcol in label_rowcols:
                row, column = rowcol
                if rowcol in self.top_scores:
                    raw_top_score = self.top_scores[rowcol]
                    score_text = f'{round(raw_top_score*100)}'
                    score_shadow_offset = score_offset - 0.5
                    # Text 'shadow', since this seems easier to accomplish than
                    # an outline, and is almost as good.
                    ax.annotate(
                        score_text, (column, row),
                        color=shadow_color,
                        fontsize=12,
                        # https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text.set_fontweight
                        fontweight='semibold',
                        # textcoords in offset units allow the text to be a
                        # consistent distance from the point marker, regardless
                        # of image scale or zoom level.
                        xytext=(score_shadow_offset, score_shadow_offset),
                        textcoords='offset points',
                    )
                    # Main text
                    ax.annotate(
                        score_text, (column, row),
                        color=colors[label],
                        fontsize=12,
                        fontweight='semibold',
                        xytext=(score_offset, score_offset),
                        textcoords='offset points',
                    )

        from matplotlib.legend_handler import HandlerTuple
        ax.legend(
            list(zip(shadow_scatters, main_scatters)),
            [self.label_ids_to_names[str(label_id)]
             for label_id in unique_top_labels],
            # Put the legend outside the plot, to the right
            loc='center left', bbox_to_anchor=(1, 0.5),
            # Spacing between the legend items
            labelspacing=1,
            handler_map={tuple: HandlerTuple(ndivide=None)},
        )

        fig.set_size_inches(11, 7)
        plt.show()
