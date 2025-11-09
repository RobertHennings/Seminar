from typing import Dict, List
import logging
import os
import ast
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.io import to_html
from scipy import stats
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf

"""
This file contains the main class for creating all graphs used and produced for the seminar
project.
"""

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# import config settings with static global variables
try:
    from . import config as cfg
except:
    import config as cfg

FONT=cfg.FONT
PLOT_BGCOLOR=cfg.PLOT_BGCOLOR
PAPER_BGCOLOR=cfg.PAPER_BGCOLOR
MARGIN=cfg.MARGIN
FILE_PATH = cfg.FILE_PATH

class DataGraphing(object):
    def __init__(
        self,
        font: str=FONT,
        plot_bgcolor: str=PLOT_BGCOLOR,
        paper_bgcolor: str=PAPER_BGCOLOR,
        margin: Dict[str, float]=MARGIN,
        file_path: str=FILE_PATH
        ):
        self.font = font
        self.plot_bgcolor = plot_bgcolor
        self.paper_bgcolor = paper_bgcolor
        self.margin = margin
        self.file_path = file_path


    ######################## Internal helper methods #########################
    def __check_path_existence(
        self,
        path: str
        ):
        """Internal helper method - serves as generous path existence
           checker when saving and reading of an kind of data from files
           suspected at the given location
           
           !!!!If given path does not exist it will be created!!!!

        Args:
            path (str): full path where expected data is saved, i.e.
                        i.e. /Users/Robert_Hennings/Uni/Master/Seminar/reports/figures
                    !!!!DO NOT INCLUDE THE FILENAME like: /Users/Robert_Hennings/Uni/Master/Seminar/reports/figures/figure.pdf
        """
        folder_name = path.split("/")[-1]
        path = "/".join(path.split("/")[:-1])
        # FileNotFoundError()
        # os.path.isdir()
        if folder_name not in os.listdir(path):
            logging.info(f"{folder_name} not found in path: {path}")
            folder_path = f"{path}/{folder_name}"
            os.mkdir(folder_path)
            logging.info(f"Folder: {folder_name} created in path: {path}")


    def _save_figure_as_pdf(
        self,
        fig: go.Figure,
        file_name: str=None,
        file_path: str=None,
        width: int=800,
        height: int=600,
        scale: int=2
        ) -> None:
        """Saves the given figure as a PDF file to be directly used as .pdf
           in the LaTeX presentation.

        Args:
            fig (go.Figure): The Plotly figure to save.
            file_name (str, optional): The specified filename (without file type ending). Defaults to None.
            file_path (str, optional): The specified path at which to save the graph. Defaults to None.
            width (int, optional): Custom width. Defaults to 800.
            height (int, optional): Custom height. Defaults to 600.
            scale (int, optional): Custom resolution scaling. Defaults to 2.

        Raises:
            ValueError: If file_name or file_path is not provided.
        """
        if file_path is not None:
            file_path_save = file_path
        elif self.file_path:
            file_path_save = self.file_path
        else:
            file_path_save = r"/"
        if file_path_save is not None and file_name is not None:
            full_saving_path = os.path.join(file_path_save, file_name)
            logging.info(f"Saving figure as PDF to: {full_saving_path}")
            self.__check_path_existence(path=file_path_save) # only check if the general path exists
            fig.write_image(
                file=full_saving_path,
                format="pdf",
                width=width,
                height=height,
                scale=scale,
                )
        else:
            raise ValueError(f"file_name and file_path must be provided when saving as HTML, is currently as: {file_name}, {file_path_save}")


    def _save_figure_as_html(
        self,
        fig: go.Figure,
        file_name: str=None,
        file_path: str=None,
        ) -> None:
        """Saves the given figure as a html file to be viewed better online or locally.

        Args:
            fig (go.Figure): _description_
            file_name (str, optional): _description_. Defaults to None.
            file_path (str, optional): _description_. Defaults to None.

        Raises:
            ValueError: If file_name or file_path is not provided.
        """
        if file_path is not None:
            file_path_save = file_path
        elif self.file_path:
            file_path_save = self.file_path
        else:
            file_path_save = r"/"
        if file_path_save is not None and file_name is not None:
            full_saving_path = os.path.join(file_path_save, file_name)
            logging.info(f"Saving figure as HTML to: {full_saving_path}")
            self.__check_path_existence(path=file_path_save)  # only check if the general path exists
            fig.write_html(
                file=full_saving_path
                )
        else:
            raise ValueError(f"file_name and file_path must be provided when saving as HTML, is currently as: {file_name}, {file_path_save}")


    ######################## Public methods #########################
    def lighten_color(
        self,
        hex_color: str,
        factor: float
        ) -> str:
        """Lightens the provided hex_color by a specified factor to create
           a full custom diverging colorscale with custom steps.

        Args:
            hex_color (str): The color to be lightened up.
            factor (float): The factor by which the provided hex color should be lightened up.

        Returns:
            str: Lightened color.
        """
        rgb = np.array(mcolors.to_rgb(hex_color))
        light_rgb = 1 - (1 - rgb) * factor  # move toward white
        return mcolors.to_hex(light_rgb)


    def create_custom_diverging_colorscale(
        self,
        start_hex: str,
        end_hex: str,
        steps: int = 5,
        center_color: str = "#ffffff",
        lightening_factor: float = 0.6
        ) -> List[[float, str]]:
        """Returns a Plotly-style custom diverging continuous color scale from start to end color.

        Args:
            start_hex (str): Hex color for the low end (e.g., '#ff0000')
            end_hex (str): Hex color for the high end (e.g., '#0000ff')
            steps (int, optional): Number of gradient steps toward the center for each side. Defaults to 5.
            center_color (str, optional): divergence midpoint (white). Defaults to "#ffffff".
            lightening_factor (float, optional): Value < 1 to determine how much each step lightens. Defaults to 0.6.

        Returns:
            List[[float, str]]: _description_
        """
        # Generate lightened steps from start -> white
        left_colors = [
            self.lighten_color(start_hex, lightening_factor ** (steps - i - 1))
            for i in range(steps)
        ][::-1]

        # Generate lightened steps from end -> white
        right_colors = [
            self.lighten_color(end_hex, lightening_factor ** i)
            for i in range(steps)
        ][::-1]

        # Combine with positions from 0 to 1
        total = 2 * steps + 1
        full_colors = left_colors + [center_color] + right_colors
        scale = [[i / (total - 1), c] for i, c in enumerate(full_colors)]

        return scale


    def get_fig_consumption_production_oil_gas(
        self,
        data: pd.DataFrame,
        variables: List[str],
        secondary_y_variables: List[str],
        title: str,
        x_axis_title: str,
        y_axis_title: str,
        secondary_y_axis_title: str,
        color_mapping_dict: Dict[str, str],
        num_years_interval_x_axis: int=5,
        margin_dict: Dict[str, float]=None,
        showlegend: bool=False,
        save_fig: bool=False,
        file_name: str=None,
        file_path: str=None,
        width: int = 800,
        height: int = 600,
        scale: int = 2
        ) -> go.Figure:
        """Creates the figure for oil and gas consumption and production
           with two y-axes and annotations for each variable at the end of
           the time series.

        Args:
            data (pd.DataFrame): _description_
            variables (List[str]): _description_
            secondary_y_variables (List[str]): _description_
            title (str): _description_
            x_axis_title (str): _description_
            y_axis_title (str): _description_
            secondary_y_axis_title (str): _description_
            color_mapping_dict (Dict[str, str]): _description_
            num_years_interval_x_axis (int, optional): _description_. Defaults to 5.
            showlegend (bool, optional): _description_. Defaults to False.
            save_fig (bool, optional): _description_. Defaults to False.
            file_name (str, optional): _description_. Defaults to None.
            file_path (str, optional): _description_. Defaults to None.
            width (int, optional): _description_. Defaults to 800.
            height (int, optional): _description_. Defaults to 600.
            scale (int, optional): _description_. Defaults to 2.

        Raises:
            ValueError: If the figure should be saved but no file_name or
                        file_path is provided.

        Returns:
            go.Figure: The consumption or production figure.
        """
        if margin_dict is not None:
            margins = margin_dict
        else:
            margins = self.margin
        if file_path is not None:
            file_path_save = file_path
        elif self.file_path:
            file_path_save = self.file_path
        else:
            file_path_save = r"/"

        fig = go.Figure()
        if not data.empty:
            for variable in variables:
                product_data = data[variable]
                product_data = pd.to_numeric(data[variable], errors='coerce')
                fig.add_trace(
                    go.Scatter(
                        x=product_data.index.strftime('%Y-%m-%d'),
                        y=product_data,
                        mode='lines+markers',
                        name=variable,
                        line=dict(color=color_mapping_dict.get(variable, "black"))  # Use the mapped color
                    )
                )
            # Add traces for the secondary y-axis variables
            for variable in secondary_y_variables:
                product_data = data[variable]
                product_data = pd.to_numeric(data[variable], errors='coerce')
                fig.add_trace(
                    go.Scatter(
                        x=product_data.index.strftime('%Y-%m-%d'),
                        y=product_data,
                        mode='lines+markers',
                        name=f"{variable} (r.h.)",
                        yaxis="y2",
                        line=dict(color=color_mapping_dict.get(variable, "black"))  # Use the mapped color
                    )
                )
            # Add annotations for each variable
            annotations = []
            for variable in variables + secondary_y_variables:
                product_data = data[variable]
                product_data = pd.to_numeric(data[variable], errors='coerce')
                if not product_data.empty:
                    # Get the last data point for the variable
                    x_last = product_data.index[-1] + pd.DateOffset(years=1)
                    # Convert to Python datetime for Plotly
                    x_last = pd.Timestamp(x_last).to_pydatetime()
                    y_last = product_data.iloc[-1]
                    ay = 0
                    if "World" in variable:
                        # y_last = y_last + 0.08 * y_last
                        y_last = y_last + 200
                    annotations.append(
                        dict(
                            x=x_last,
                            y=y_last,
                            xref="x",
                            yref="y" if variable not in secondary_y_variables else "y2",
                            text=variable,
                            showarrow=True,
                            arrowhead=1,
                            ax=20,  # Offset for the annotation line
                            ay=ay,
                            font=dict(size=10, color=color_mapping_dict.get(variable, "black")),
                            arrowcolor=color_mapping_dict.get(variable, "black"),
                            arrowwidth=1.0
                        )
                    )
            tickvals = pd.date_range(
                start=pd.Timestamp(year=data.index.min().year, month=1, day=1),
                end=data.index.max(),
                freq=f"{num_years_interval_x_axis}YS"
            ).to_pydatetime()
            # Update the layout
            fig.update_layout(
                title={
                    "text": title,
                    "x": 0.5,  # Center the title
                    "xanchor": "center",  # Anchor the title at the center
                    "yanchor": "top"  # Anchor the title at the top
                },
                xaxis=dict(
                    title=x_axis_title,
                    gridcolor="lightgrey",
                    tickformat="%Y",
                    tickvals=tickvals
                ),
                yaxis=dict(
                    title=y_axis_title,
                    side="left",
                    gridcolor="lightgrey"  # Set y-axis grid lines to light grey
                ),
                yaxis2=dict(
                    title=secondary_y_axis_title,
                    overlaying="y",
                    side="right",
                    gridcolor="lightgrey"
                ),
                font=self.font,
                margin=margins,
                plot_bgcolor=self.plot_bgcolor,
                paper_bgcolor=self.paper_bgcolor,
                showlegend=showlegend,  # Disable the legend
                annotations=annotations  # Add the annotations
            )
        if save_fig:
            if file_name is None or file_path_save is None:
                raise ValueError(f"file_name and file_path must be provided when save_fig is True, is currently as: {file_name}, {file_path_save}")
            else:
                file_name = file_name if file_name.endswith(".pdf") else f"{file_name}.pdf"
                logging.info(f"Saving figure as PDF to: {file_path_save} with name: {file_name}")
                self._save_figure_as_pdf(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save,
                    width=width,
                    height=height,
                    scale=scale
                )
                file_name = file_name.replace(".pdf", ".html") if file_name.endswith(".pdf") else f"{file_name}.html"
                logging.info(f"Saving figure as HTML to: {file_path_save} with name: {file_name}")
                self._save_figure_as_html(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save
                )

        return fig


    def get_combined_production_consumption_graph(
        self,
        subplot_titles: set[str],
        title: str,
        num_years_interval_x_axis: int=5,
        x_axis_title: str="Year",
        secondary_y_variable: str="World",
        rows: int=2,
        cols: int=1,
        shared_xaxes: bool=False,
        vertical_spacing: float=0.25,
        specs: List[List[Dict[str, str]]]=[[{"secondary_y": True}], [{"secondary_y": True}]],
        fig_production: go.Figure=None,
        fig_consumption: go.Figure=None,
        margin_dict: Dict[str, float]=None,
        showlegend: bool=False,
        save_fig: bool=False,
        file_name: str=None,
        file_path: str=None,
        width: int=800,
        height: int=600,
        scale: int=2
        ) -> go.Figure:
        if margin_dict is not None:
            margins = margin_dict
        else:
            margins = self.margin
        if file_path is not None:
            file_path_save = file_path
        elif self.file_path:
            file_path_save = self.file_path
        else:
            file_path_save = r"/"
        fig_consumption_production_combine = make_subplots(
            rows=rows,
            cols=cols,
            shared_xaxes=shared_xaxes,
            vertical_spacing=vertical_spacing,
            subplot_titles=subplot_titles,
            specs=specs
        )
        for trace in fig_consumption.data:
            trace.x = pd.to_datetime(trace.x).to_pydatetime()
            fig_consumption_production_combine.add_trace(trace, row=1, col=1, secondary_y=(secondary_y_variable in trace.name))
            # Place label near the last point
            if trace.x.size == 0 or trace.y.size == 0:
                continue
            xval = pd.to_datetime(trace.x[-1]) + pd.DateOffset(years=1)
            yval = trace.y[-1]
            fig_consumption_production_combine.add_annotation(
                x=xval, y=yval,
                text=trace.name,
                showarrow=False,
                xanchor="left", yanchor="middle",
                font=dict(color=trace.line.color),
                row=1, col=1,  # Annotation in row 1
                secondary_y=(secondary_y_variable in trace.name)
            )

        for trace in fig_production.data:
            trace.x = pd.to_datetime(trace.x).to_pydatetime()
            fig_consumption_production_combine.add_trace(trace, row=2, col=1, secondary_y=(secondary_y_variable in trace.name))
            if trace.x.size == 0 or trace.y.size == 0:
                continue
            xval = pd.to_datetime(trace.x[-1]) + pd.DateOffset(years=1)
            yval = trace.y[-1]
            if "World" in trace.name:
                yval += 1000
            fig_consumption_production_combine.add_annotation(
                x=xval, y=yval,
                text=trace.name,
                showarrow=False,
                xanchor="left", yanchor="middle",
                font=dict(color=trace.line.color),
                row=2, col=1,  # Annotation in row 2
                secondary_y=(secondary_y_variable in trace.name)
            )

        fig_consumption_production_combine.update_yaxes(title_text="Consumption (TWh)", row=1, col=1, secondary_y=False, gridcolor="lightgrey")
        fig_consumption_production_combine.update_yaxes(title_text="World Consumption (TWh)", row=1, col=1, secondary_y=True, gridcolor="lightgrey")
        fig_consumption_production_combine.update_yaxes(title_text="Production (TWh)", row=2, col=1, secondary_y=False, gridcolor="lightgrey")
        fig_consumption_production_combine.update_yaxes(title_text="World Production (TWh)", row=2, col=1, secondary_y=True, gridcolor="lightgrey")

        # Update the gridcolor for the main x-axis
        data = fig_production.data[0].x
        data = pd.to_datetime(data)
        tickvals = pd.date_range(
                start=pd.Timestamp(year=data.min().year, month=1, day=1),
                end=pd.Timestamp(year=data.max().year, month=1, day=1),
                freq=f"{num_years_interval_x_axis}YS"
            ).to_pydatetime()
        fig_consumption_production_combine.update_xaxes(title_text=x_axis_title, gridcolor="lightgrey", tickformat="%Y", tickvals=tickvals, row=1, col=1)
        fig_consumption_production_combine.update_xaxes(title_text=x_axis_title, gridcolor="lightgrey", tickformat="%Y", tickvals=tickvals, row=2, col=1)

        fig_consumption_production_combine.update_layout(
            title={
                "text": title,
                "x": 0.5,  # Center the title
                "xanchor": "center",  # Anchor the title at the center
                "yanchor": "top",  # Anchor the title at the top
                "y": 0.95  # Adjust the vertical position of the title
            },
            font=self.font,
            margin=margins,
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.paper_bgcolor,
            showlegend=showlegend
        )
        if save_fig:
            if file_name is None or file_path_save is None:
                raise ValueError(f"file_name and file_path must be provided when save_fig is True, is currently as: {file_name}, {file_path_save}")
            else:
                file_name = file_name if file_name.endswith(".pdf") else f"{file_name}.pdf"
                logging.info(f"Saving figure as PDF to: {file_path_save} with name: {file_name}")
                self._save_figure_as_pdf(
                    fig=fig_consumption_production_combine,
                    file_name=file_name,
                    file_path=file_path_save,
                    width=width,
                    height=height,
                    scale=scale
                )
                file_name = file_name.replace(".pdf", ".html") if file_name.endswith(".pdf") else f"{file_name}.html"
                logging.info(f"Saving figure as HTML to: {file_path_save} with name: {file_name}")
                self._save_figure_as_html(
                    fig=fig_consumption_production_combine,
                    file_name=file_name,
                    file_path=file_path_save
                )

        return fig_consumption_production_combine


    def get_fig_open_interest(
        self,
        data: pd.DataFrame,
        variables: List[str],
        secondary_y_variables: List[str],
        title: str,
        x_axis_title: str,
        y_axis_title: str,
        secondary_y_axis_title: str,
        color_mapping_dict: Dict[str, str],
        num_years_interval_x_axis: int=5,
        margin_dict: Dict[str, float]=None,
        showlegend: bool=False,
        save_fig: bool=False,
        file_name: str=None,
        file_path: str=None,
        width: int = 800,
        height: int = 600,
        scale: int = 2
        ) -> go.Figure:
        if margin_dict is not None:
            margins = margin_dict
        else:
            margins = self.margin
        if file_path is not None:
            file_path_save = file_path
        elif self.file_path:
            file_path_save = self.file_path
        else:
            file_path_save = r"/"

        fig_oi = go.Figure()
        # Add traces for the primary y-axis variable
        if not data.empty:
            for variable in variables:
                product_data = data[variable]
                product_data = pd.to_numeric(data[variable], errors='coerce')
                fig_oi.add_trace(
                    go.Scatter(
                        x=product_data.index.strftime('%Y-%m-%d'),
                        y=product_data,
                        mode='lines+markers',
                        name=variable,
                        line=dict(color=color_mapping_dict.get(variable, "black"))  # Use the mapped color
                    )
                )
            # Add traces for the secondary y-axis variables
            for variable in secondary_y_variables:
                product_data = data[variable]
                product_data = pd.to_numeric(data[variable], errors='coerce')
                fig_oi.add_trace(
                    go.Scatter(
                        x=product_data.index.strftime('%Y-%m-%d'),
                        y=product_data,
                        mode='lines+markers',
                        name=f"{variable} (r.h.)",
                        yaxis="y2",
                        line=dict(color=color_mapping_dict.get(variable, "black"))  # Use the mapped color
                    )
                )
        # Add annotations for each variable
        annotations = []
        for variable in variables + secondary_y_variables:
            product_data = data[variable]
            product_data = pd.to_numeric(data[variable], errors='coerce')
            if not product_data.empty:
                # Get the last data point for the variable
                x_last = product_data.index[-1] + pd.DateOffset(days=1)
                # Convert to Python datetime for Plotly
                x_last = pd.Timestamp(x_last).to_pydatetime()
                y_last = product_data.iloc[-1]
                if variable == "Oil Open Interest USD" or variable == "Gas Open Interest USD":
                    x_last = product_data.index[-1] + pd.DateOffset(days=80)
                    x_last = pd.Timestamp(x_last).to_pydatetime()
                    y_last = product_data.iloc[-1] + 0.2 * product_data.iloc[-1]
                annotations.append(
                    dict(
                        x=x_last,
                        y=y_last,
                        xref="x",
                        yref="y" if variable not in secondary_y_variables else "y2",
                        text=variable,
                        showarrow=True,
                        arrowhead=1,
                        ax=20,  # Offset for the annotation line
                        ay=0,
                        font=dict(size=10, color=color_mapping_dict.get(variable, "black")),
                        arrowcolor=color_mapping_dict.get(variable, "black"),
                        arrowwidth=1.0
                    )
                )

        tickvals = pd.date_range(
            start=pd.Timestamp(year=data.index.min().year, month=1, day=1),
            end=data.index.max(),
            freq=f"{num_years_interval_x_axis}YS"
        ).to_pydatetime()
        # Update the layout
        fig_oi.update_layout(
        title={
            "text": title,
            "x": 0.5,  # Center the title
            "xanchor": "center",  # Anchor the title at the center
            "yanchor": "top"  # Anchor the title at the top
        },
        xaxis=dict(
            title=x_axis_title,
            gridcolor="lightgrey",
            tickformat="%Y",
            tickvals=tickvals
        ),
        yaxis=dict(
            title=y_axis_title,
            side="left",
            gridcolor="lightgrey"  # Set y-axis grid lines to light grey
        ),
        yaxis2=dict(
            title=secondary_y_axis_title,
            overlaying="y",
            side="right",
            gridcolor="lightgrey"
        ),
        font=self.font,
        margin=margins,
        plot_bgcolor=self.plot_bgcolor,
        paper_bgcolor=self.paper_bgcolor,
        showlegend=showlegend,  # Disable the legend
        annotations=annotations  # Add the annotations
        )
        if save_fig:
            if file_name is None or file_path_save is None:
                raise ValueError(f"file_name and file_path must be provided when save_fig is True, is currently as: {file_name}, {file_path_save}")
            else:
                file_name = file_name if file_name.endswith(".pdf") else f"{file_name}.pdf"
                logging.info(f"Saving figure as PDF to: {file_path_save} with name: {file_name}")
                self._save_figure_as_pdf(
                    fig=fig_oi,
                    file_name=file_name,
                    file_path=file_path_save,
                    width=width,
                    height=height,
                    scale=scale
                )
                file_name = file_name.replace(".pdf", ".html") if file_name.endswith(".pdf") else f"{file_name}.html"
                logging.info(f"Saving figure as HTML to: {file_path_save} with name: {file_name}")
                self._save_figure_as_html(
                    fig=fig_oi,
                    file_name=file_name,
                    file_path=file_path_save
                )
        return fig_oi


    def get_combined_open_interest_graph(
        self,
        subplot_titles: set[str],
        title: str,
        num_years_interval_x_axis: int=5,
        x_axis_title: str="Date",
        secondary_y_variable: str="World",
        rows: int=2,
        cols: int=1,
        shared_xaxes: bool=False,
        vertical_spacing: float=0.25,
        specs: List[List[Dict[str, str]]]=[{"secondary_y": True}, {"secondary_y": True}],
        fig_oil_oi: go.Figure=None,
        fig_gas_oi: go.Figure=None,
        margin_dict: Dict[str, float]=None,
        showlegend: bool=False,
        save_fig: bool=False,
        file_name: str=None,
        file_path: str=None,
        width: int=800,
        height: int=600,
        scale: int=2
        ) -> go.Figure:
        if margin_dict is not None:
            margins = margin_dict
        else:
            margins = self.margin
        if file_path is not None:
            file_path_save = file_path
        elif self.file_path:
            file_path_save = self.file_path
        else:
            file_path_save = r"/"

        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_xaxes=shared_xaxes,
            vertical_spacing=vertical_spacing,
            subplot_titles=subplot_titles,
            specs=specs
        )

        # Add traces for oil consumption (row 1)
        for trace in fig_oil_oi.data:
            # trace.x = pd.to_datetime(trace.x)
            fig.add_trace(trace, row=1, col=1, secondary_y=("Oil Open Interest USD" in trace.name))
            # Place label near the last point
            if trace.x.size == 0 or trace.y.size == 0:
                continue
            xval = trace.x[-1]# + pd.DateOffset(days=1)
            yval = trace.y[-1]
            if "Oil Open Interest USD" in trace.name:
                yval += 0.2 * yval
            fig.add_annotation(
                x=xval, y=yval,
                text=trace.name,
                showarrow=False,
                xanchor="left", yanchor="middle",
                font=dict(color=trace.line.color),
                row=1, col=1,  # Annotation in row 1
                secondary_y=("Oil Open Interest USD" in trace.name)
            )

        # Add traces for oil production (row 2)
        for trace in fig_gas_oi.data:
            # trace.x = pd.to_datetime(trace.x)
            fig.add_trace(trace, row=2, col=1, secondary_y=("Gas Open Interest USD" in trace.name))
            # Place label near the last point
            if trace.x.size == 0 or trace.y.size == 0:
                continue
            xval = trace.x[-1]# + pd.DateOffset(days=1)
            yval = trace.y[-1]
            if "Gas Open Interest USD" in trace.name:
                yval += 0.2 * yval
            fig.add_annotation(
                x=xval, y=yval,
                text=trace.name,
                showarrow=False,
                xanchor="left", yanchor="middle",
                font=dict(color=trace.line.color),
                row=2, col=1,  # Annotation in row 2
                secondary_y=("Gas Open Interest USD" in trace.name)
            )

        fig.update_yaxes(title_text="Open Interest (number of contracts)", row=1, col=1, secondary_y=False, gridcolor="lightgrey")
        fig.update_yaxes(title_text="Open Interest (in USD)", row=1, col=1, secondary_y=True, gridcolor="lightgrey")
        fig.update_yaxes(title_text="Open Interest (number of contracts)", row=2, col=1, secondary_y=False, gridcolor="lightgrey")
        fig.update_yaxes(title_text="Open Interest (in USD)", row=2, col=1, secondary_y=True, gridcolor="lightgrey")

        data = fig_gas_oi.data[0].x
        data = pd.to_datetime(data)
        tickvals = pd.date_range(
                start=pd.Timestamp(year=data.min().year, month=1, day=1),
                end=data.max(),
                freq=f"{num_years_interval_x_axis}YS"
            ).to_pydatetime()
        fig.update_xaxes(title_text=x_axis_title, gridcolor="lightgrey", tickformat="%Y", tickvals=tickvals, row=1, col=1)
        fig.update_xaxes(title_text=x_axis_title, gridcolor="lightgrey", tickformat="%Y", tickvals=tickvals, row=2, col=1)
        fig.update_layout(
            title={
                "text": title,
                "x": 0.5,  # Center the title
                "xanchor": "center",  # Anchor the title at the center
                "yanchor": "top",  # Anchor the title at the top
                "y": 0.95  # Adjust the vertical position of the title
            },
            font=self.font,
            margin=margins,
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.paper_bgcolor,
            showlegend=showlegend
        )
        if save_fig:
            if file_name is None or file_path_save is None:
                raise ValueError(f"file_name and file_path must be provided when save_fig is True, is currently as: {file_name}, {file_path_save}")
            else:
                file_name = file_name if file_name.endswith(".pdf") else f"{file_name}.pdf"
                logging.info(f"Saving figure as PDF to: {file_path_save} with name: {file_name}")
                self._save_figure_as_pdf(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save,
                    width=width,
                    height=height,
                    scale=scale
                )
                file_name = file_name.replace(".pdf", ".html") if file_name.endswith(".pdf") else f"{file_name}.html"
                logging.info(f"Saving figure as HTML to: {file_path_save} with name: {file_name}")
                self._save_figure_as_html(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save
                )
        return fig


    def get_fig_inflation_contribution_usa(
        self,
        data: pd.DataFrame,
        data_pct: pd.DataFrame,
        cpi_color: str,
        variables: List[str],
        secondary_y_variables: List[str],
        title: str,
        x_axis_title: str,
        y_axis_title: str,
        color_mapping_dict: Dict[str, str],
        num_years_interval_x_axis: int=5,
        margin_dict: Dict[str, float]=None,
        showlegend: bool=False,
        save_fig: bool=False,
        file_name: str=None,
        file_path: str=None,
        width: int = 800,
        height: int = 600,
        scale: int = 2
        ) -> go.Figure:
        if margin_dict is not None:
            margins = margin_dict
        else:
            margins = self.margin
        if file_path is not None:
            file_path_save = file_path
        elif self.file_path:
            file_path_save = self.file_path
        else:
            file_path_save = r"/"

        fig = go.Figure()
        # Add each bar component to the stack
        for component in variables:
            fig.add_trace(go.Bar(
                x=data.index.to_pydatetime(),
                y=data[component],
                name=component,
                showlegend=False,
                marker_color=color_mapping_dict.get(component, "black")
            ))

        # Overlay the headline CPI as a line
        fig.add_trace(go.Scatter(
            x=data.index.to_pydatetime(),
            y=data_pct['Headline'].reindex(data.index),
            name='Headline CPI',
            mode='lines+markers',
            showlegend=False,
            line=dict(color=cpi_color, width=2)
        ))
        # Add annotations for each variable
        annotations = []
        for variable in variables + ["Headline"]:
            product_data = data_pct[variable]
            if not product_data.empty:
                # Get the last data point for the variable
                x_last = product_data.index[-1] + pd.DateOffset(days=12)
                x_last = pd.Timestamp(x_last).to_pydatetime()
                y_last = product_data[-1]
                annotations.append(
                    dict(
                        x=x_last,
                        y=y_last,
                        xref="x",
                        yref="y" if variable not in secondary_y_variables else "y2",
                        text=variable,
                        showarrow=True,
                        arrowhead=1,
                        ax=20,  # Offset for the annotation line
                        ay=0,
                        font=dict(size=10, color=color_mapping_dict.get(variable, "black")),
                        arrowcolor=color_mapping_dict.get(variable, "black"),
                        arrowwidth=1.0
                    )
                )
        tickvals = pd.date_range(
            start=pd.Timestamp(year=data.index.min().year, month=1, day=1),
            end=data.index.max(),
            freq=f"{num_years_interval_x_axis}YS"
            ).to_pydatetime()
        # Set stacked bar mode
        fig.update_layout(
            barmode='stack',
            title={
                "text": title,
                "x": 0.5,  # Center the title
                "xanchor": "center",  # Anchor the title at the center
                "yanchor": "top",  # Anchor the title at the top
                "y": 0.95  # Adjust the vertical position of the title
            },
            xaxis=dict(
                title=x_axis_title,
                tickformat='%Y',
                gridcolor="lightgrey",  # Set x-axis grid lines to light grey
                tickvals=tickvals
                ),
            yaxis=dict(
                title=y_axis_title,
                gridcolor="lightgrey"  # Set y-axis grid lines to light grey
                ),
            font=self.font,
            margin=margins,
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.paper_bgcolor,
            annotations=annotations,
            showlegend=showlegend
        )
        if save_fig:
            if file_name is None or file_path_save is None:
                raise ValueError(f"file_name and file_path must be provided when save_fig is True, is currently as: {file_name}, {file_path_save}")
            else:
                file_name = file_name if file_name.endswith(".pdf") else f"{file_name}.pdf"
                logging.info(f"Saving figure as PDF to: {file_path_save} with name: {file_name}")
                self._save_figure_as_pdf(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save,
                    width=width,
                    height=height,
                    scale=scale
                )
                file_name = file_name.replace(".pdf", ".html") if file_name.endswith(".pdf") else f"{file_name}.html"
                logging.info(f"Saving figure as HTML to: {file_path_save} with name: {file_name}")
                self._save_figure_as_html(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save
                )
        return fig


    def get_fig_inflation_contribution_euro_area(
        self,
        data: pd.DataFrame,
        cpi_color: str,
        variables: List[str],
        secondary_y_variables: List[str],
        title: str,
        x_axis_title: str,
        y_axis_title: str,
        color_mapping_dict: Dict[str, str],
        num_years_interval_x_axis: int=5,
        margin_dict: Dict[str, float]=None,
        showlegend: bool=False,
        save_fig: bool=False,
        file_name: str=None,
        file_path: str=None,
        width: int = 800,
        height: int = 600,
        scale: int = 2
        ) -> go.Figure:
        if margin_dict is not None:
            margins = margin_dict
        else:
            margins = self.margin
        if file_path is not None:
            file_path_save = file_path
        elif self.file_path:
            file_path_save = self.file_path
        else:
            file_path_save = r"/"

        fig = go.Figure()
        # Add each bar component to the stack
        for component in variables:
            fig.add_trace(go.Bar(
                x=data.index.to_pydatetime(),
                y=data[component],
                name=component,
                showlegend=False,
                marker_color=color_mapping_dict.get(component, "black")
            ))

        # Overlay the headline CPI as a line
        fig.add_trace(go.Scatter(
            x=data.index.to_pydatetime(),
            y=data['Headline'].reindex(data.index),
            name='Headline',
            mode='lines+markers',
            showlegend=False,
            line=dict(color=cpi_color, width=2)
        ))
        # Add annotations for each variable
        annotations = []
        for variable in variables + ["Headline"]:
            product_data = data[variable]
            if not product_data.empty:
                # Get the last data point for the variable
                x_last = product_data.index[-1] + pd.DateOffset(days=12)
                x_last = pd.Timestamp(x_last).to_pydatetime()
                y_last = product_data[-1]
                annotations.append(
                    dict(
                        x=x_last,
                        y=y_last,
                        xref="x",
                        yref="y" if variable not in secondary_y_variables else "y2",
                        text=variable,
                        showarrow=True,
                        arrowhead=1,
                        ax=20,  # Offset for the annotation line
                        ay=0,
                        font=dict(size=10, color=color_mapping_dict.get(variable, "black")),
                        arrowcolor=color_mapping_dict.get(variable, "black"),
                        arrowwidth=1.0
                    )
                )
        tickvals = pd.date_range(
            start=pd.Timestamp(year=data.index.min().year, month=1, day=1),
            end=data.index.max(),
            freq=f"{num_years_interval_x_axis}YS"
            ).to_pydatetime()
        # Set stacked bar mode
        fig.update_layout(
            barmode='stack',
            title={
                "text": title,
                "x": 0.5,  # Center the title
                "xanchor": "center",  # Anchor the title at the center
                "yanchor": "top",  # Anchor the title at the top
                "y": 0.95  # Adjust the vertical position of the title
            },
            xaxis=dict(
                title=x_axis_title,
                tickformat='%Y',
                gridcolor="lightgrey",  # Set x-axis grid lines to light grey
                tickvals=tickvals
            ),
            yaxis=dict(
                title=y_axis_title,
                gridcolor="lightgrey"  # Set y-axis grid lines to light grey
                ),
            font=self.font,
            margin=margins,
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.paper_bgcolor,
            annotations=annotations,  # Add the annotations
            showlegend=showlegend
        )
        if save_fig:
            if file_name is None or file_path_save is None:
                raise ValueError(f"file_name and file_path must be provided when save_fig is True, is currently as: {file_name}, {file_path_save}")
            else:
                file_name = file_name if file_name.endswith(".pdf") else f"{file_name}.pdf"
                logging.info(f"Saving figure as PDF to: {file_path_save} with name: {file_name}")
                self._save_figure_as_pdf(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save,
                    width=width,
                    height=height,
                    scale=scale
                )
                file_name = file_name.replace(".pdf", ".html") if file_name.endswith(".pdf") else f"{file_name}.html"
                logging.info(f"Saving figure as HTML to: {file_path_save} with name: {file_name}")
                self._save_figure_as_html(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save
                )
        return fig


    def get_fig_deviations_ppp(
        self,
        data: pd.DataFrame,
        variables: List[str],
        secondary_y_variables: List[str],
        title: str,
        secondary_y_axis_title: str,
        color_discrete_sequence: List[str],
        x_axis_title: str,
        y_axis_title: str,
        color_mapping_dict: Dict[str, str],
        num_years_interval_x_axis: int=5,
        margin_dict: Dict[str, float]=None,
        showlegend: bool=False,
        save_fig: bool=False,
        file_name: str=None,
        file_path: str=None,
        width: int = 800,
        height: int = 600,
        scale: int = 2
        ) -> go.Figure:
        if margin_dict is not None:
            margins = margin_dict
        else:
            margins = self.margin
        if file_path is not None:
            file_path_save = file_path
        elif self.file_path:
            file_path_save = self.file_path
        else:
            file_path_save = r"/"

        fig = go.Figure()

        for i, var in enumerate(variables):
            if secondary_y_variables and var in secondary_y_variables:
                continue  # Skip variables meant for the secondary y-axis
            fig.add_trace(
                go.Scatter(
                    x=data.index.to_pydatetime(),
                    y=data[var],
                    mode='lines',
                    name=str(var),
                    opacity=1.0,
                    showlegend=False,
                    line=dict(
                        color=color_mapping_dict.get(var, color_discrete_sequence[i % len(color_discrete_sequence)]),
                        width=2.0
                    )
                )
            )
        # Add a horizontal line at y=0
        fig.add_shape(
            type="line",
            x0=pd.Timestamp(data.index.min()).to_pydatetime(),
            y0=0,
            x1=pd.Timestamp(data.index.max()).to_pydatetime(),
            y1=0,
            line=dict(color="black", width=2, dash="dash"),
            xref="x",
            yref="y"
        )
        # Plot variables on the secondary y-axis
        if secondary_y_variables:
            for i, var in enumerate(secondary_y_variables):
                if var not in data.columns:
                    logging.info(f"Variable '{var}' not found in data columns. Skipping.")
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=data.index.to_pydatetime(),
                        y=data[var],
                        mode='lines',
                        name=f"{var} (r.h.)",
                        opacity=1.0,
                        showlegend=False,
                        line=dict(
                            color=color_mapping_dict.get(var, color_discrete_sequence[i % len(color_discrete_sequence)]),
                            width=2.0
                        ),
                        yaxis="y2"  # Assign to secondary y-axis
                    )
                )
        # Add annotations for each variable
        annotations = []
        for variable in variables + secondary_y_variables:
            product_data = data[variable]
            if not product_data.empty:
                # Get the last data point for the variable
                x_last = pd.Timestamp(product_data.index[-1] + pd.DateOffset(days=12)).to_pydatetime()
                y_last = product_data[-1]
                annotations.append(
                    dict(
                        x=x_last,
                        y=y_last,
                        xref="x",
                        yref="y" if variable not in secondary_y_variables else "y2",
                        text=variable,
                        showarrow=True,
                        arrowhead=1,
                        ax=20,  # Offset for the annotation line
                        ay=0,
                        font=dict(size=10, color=color_mapping_dict.get(variable, "black")),
                        arrowcolor=color_mapping_dict.get(variable, "black"),
                        arrowwidth=1.0
                    )
                )
        tickvals = pd.date_range(
            start=pd.Timestamp(year=data.index.min().year, month=1, day=1),
            end=data.index.max(),
            freq=f"{num_years_interval_x_axis}YS"
            ).to_pydatetime()
        # Update layout with dual y-axes
        fig.update_layout(
            title={
                "text": title,
                "x": 0.5,  # Center the title
                "xanchor": "center",  # Anchor the title at the center
                "yanchor": "top",  # Anchor the title at the top
                "y": 0.95  # Adjust the vertical position of the title
            },
            xaxis=dict(
                title=x_axis_title,
                tickformat='%Y',
                gridcolor="lightgrey",  # Set x-axis grid lines to light grey
                tickvals=tickvals
                ),
            yaxis=dict(
                title=y_axis_title,
                gridcolor="lightgrey"  # Set y-axis grid lines to light grey
                ),
            yaxis2=dict(
                title=secondary_y_axis_title,
                overlaying="y",  # Overlay on the primary y-axis
                side="right",  # Place on the right side
                gridcolor="lightgrey",
                ),
            font=self.font,
            margin=margins,
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.paper_bgcolor,
            annotations=annotations,  # Add the annotations
            showlegend=showlegend
        )
        if save_fig:
            if file_name is None or file_path_save is None:
                raise ValueError(f"file_name and file_path must be provided when save_fig is True, is currently as: {file_name}, {file_path_save}")
            else:
                file_name = file_name if file_name.endswith(".pdf") else f"{file_name}.pdf"
                logging.info(f"Saving figure as PDF to: {file_path_save} with name: {file_name}")
                self._save_figure_as_pdf(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save,
                    width=width,
                    height=height,
                    scale=scale
                )
                file_name = file_name.replace(".pdf", ".html") if file_name.endswith(".pdf") else f"{file_name}.html"
                logging.info(f"Saving figure as HTML to: {file_path_save} with name: {file_name}")
                self._save_figure_as_html(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save
                )
        return fig


    def get_fig_relationship_main_vars(
        self,
        data: pd.DataFrame,
        variables: List[str],
        secondary_y_variables: List[str],
        title: str,
        secondary_y_axis_title: str,
        color_discrete_sequence: List[str],
        x_axis_title: str,
        y_axis_title: str,
        color_mapping_dict: Dict[str, str],
        num_years_interval_x_axis: int=5,
        margin_dict: Dict[str, float]=None,
        showlegend: bool=False,
        save_fig: bool=False,
        file_name: str=None,
        file_path: str=None,
        width: int = 800,
        height: int = 600,
        scale: int = 2
        ) -> go.Figure:
        if margin_dict is not None:
            margins = margin_dict
        else:
            margins = self.margin
        if file_path is not None:
            file_path_save = file_path
        elif self.file_path:
            file_path_save = self.file_path
        else:
            file_path_save = r"/"

        fig = go.Figure()
        # Plot variables on the primary y-axis
        for i, var in enumerate(variables):
            if secondary_y_variables and var in secondary_y_variables:
                continue
            fig.add_trace(
                go.Scatter(
                    x=data.index.to_pydatetime(),  # <-- FIXED
                    y=data[var],
                    mode='lines',
                    name=str(var),
                    opacity=1.0,
                    showlegend=False,
                    line=dict(
                        color=color_mapping_dict.get(var, color_discrete_sequence[i % len(color_discrete_sequence)]),
                        width=2.0
                    )
                )
            )

        # Plot variables on the secondary y-axis
        if secondary_y_variables:
            for i, var in enumerate(secondary_y_variables):
                if var not in data.columns:
                    logging.info(f"Variable '{var}' not found in data columns. Skipping.")
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=data.index.to_pydatetime(),  # <-- FIXED
                        y=data[var],
                        mode='lines',
                        name=f"{var} (r.h.)",
                        opacity=1.0,
                        showlegend=False,
                        line=dict(
                            color=color_mapping_dict.get(var, color_discrete_sequence[i % len(color_discrete_sequence)]),
                            width=2.0
                        ),
                        yaxis="y2"
                    )
                )

        # Add annotations for each variable
        annotations = []
        for variable in data.columns:
            product_data = data[variable]
            if not product_data.empty:
                # Get the last data point for the variable
                x_last = pd.Timestamp(product_data.index[-1] + pd.DateOffset(days=12)).to_pydatetime()  # <-- FIXED
                y_last = product_data[-1]
                annotations.append(
                    dict(
                        x=x_last,
                        y=y_last,
                        xref="x",
                        yref="y" if variable not in secondary_y_variables else "y2",
                        text=variable,
                        showarrow=True,
                        arrowhead=1,
                        ax=20,
                        ay=0,
                        font=dict(size=10, color=color_mapping_dict.get(variable, "black")),
                        arrowcolor=color_mapping_dict.get(variable, "black"),
                        arrowwidth=1.0
                    )
                )
        tickvals = pd.date_range(
            start=pd.Timestamp(year=data.index.min().year, month=1, day=1),
            end=data.index.max(),
            freq=f"{num_years_interval_x_axis}YS"
            ).to_pydatetime()
        # Update layout with dual y-axes
        fig.update_layout(
            title={
                "text": title,
                "x": 0.5,  # Center the title
                "xanchor": "center",  # Anchor the title at the center
                "yanchor": "top",  # Anchor the title at the top
                "y": 0.95  # Adjust the vertical position of the title
            },
            xaxis=dict(
                title=x_axis_title,
                tickformat='%Y',
                tickvals=tickvals,
                gridcolor="lightgrey",  # Set x-axis grid lines to light grey
                ),
            yaxis=dict(
                title=y_axis_title,
                gridcolor="lightgrey"  # Set y-axis grid lines to light grey
                ),
            yaxis2=dict(
                title=secondary_y_axis_title,
                overlaying="y",  # Overlay on the primary y-axis
                side="right",  # Place on the right side
                gridcolor="lightgrey",
                ),
            font=self.font,
            margin=margins,
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.paper_bgcolor,
            annotations=annotations,  # Add the annotations
            showlegend=showlegend
        )
        if save_fig:
            if file_name is None or file_path_save is None:
                raise ValueError(f"file_name and file_path must be provided when save_fig is True, is currently as: {file_name}, {file_path_save}")
            else:
                file_name = file_name if file_name.endswith(".pdf") else f"{file_name}.pdf"
                logging.info(f"Saving figure as PDF to: {file_path_save} with name: {file_name}")
                self._save_figure_as_pdf(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save,
                    width=width,
                    height=height,
                    scale=scale
                )
                file_name = file_name.replace(".pdf", ".html") if file_name.endswith(".pdf") else f"{file_name}.html"
                logging.info(f"Saving figure as HTML to: {file_path_save} with name: {file_name}")
                self._save_figure_as_html(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save
                )
        return fig


    def get_fig_rolling_correlation(
        self,
        data: pd.DataFrame,
        variables: List[str],
        secondary_y_variables: List[str],
        title: str,
        secondary_y_axis_title: str,
        color_discrete_sequence: List[str],
        # x_axis_variable: str,
        # y_axis_variable: str,
        x_axis_title: str,
        y_axis_title: str,
        color_mapping_dict: Dict[str, str],
        num_years_interval_x_axis: int=5,
        margin_dict: Dict[str, float]=None,
        showlegend: bool=False,
        save_fig: bool=False,
        file_name: str=None,
        file_path: str=None,
        width: int = 800,
        height: int = 600,
        scale: int = 2
        ) -> go.Figure:
        if margin_dict is not None:
            margins = margin_dict
        else:
            margins = self.margin
        if file_path is not None:
            file_path_save = file_path
        elif self.file_path:
            file_path_save = self.file_path
        else:
            file_path_save = r"/"

        fig = go.Figure()
        # Plot variables on the primary y-axis
        for i, var in enumerate(variables):
            if secondary_y_variables and var in secondary_y_variables:
                continue  # Skip variables meant for the secondary y-axis
            fig.add_trace(
                go.Scatter(
                    x=data.index.to_pydatetime(),  # <-- FIXED
                    y=data[var],
                    mode='lines',
                    name=str(var),
                    opacity=1.0,
                    showlegend=False,
                    line=dict(
                        color=color_mapping_dict.get(var, color_discrete_sequence[i % len(color_discrete_sequence)]),
                        width=2.0
                    )
                )
            )

        # Plot variables on the secondary y-axis
        if secondary_y_variables:
            for i, var in enumerate(secondary_y_variables):
                if var not in data.columns:
                    logging.info(f"Variable '{var}' not found in data columns. Skipping.")
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=data.index.to_pydatetime(),  # <-- FIXED
                        y=data[var],
                        mode='lines',
                        name=f"{var} (r.h.)",
                        opacity=1.0,
                        showlegend=False,
                        line=dict(
                            color=color_mapping_dict.get(var, color_discrete_sequence[i % len(color_discrete_sequence)]),
                            width=2.0
                        ),
                        yaxis="y2"  # Assign to secondary y-axis
                    )
                )

        # Add annotations for each variable
        annotations = []
        for variable in data.columns:
            product_data = data[variable]
            if not product_data.empty:
                # Get the last data point for the variable
                x_last = pd.Timestamp(product_data.index[-1] + pd.DateOffset(days=12)).to_pydatetime()  # <-- FIXED
                y_last = product_data[-1]
                annotations.append(
                    dict(
                        x=x_last,
                        y=y_last,
                        xref="x",
                        yref="y" if variable not in secondary_y_variables else "y2",
                        text=variable,
                        showarrow=True,
                        arrowhead=1,
                        ax=20,  # Offset for the annotation line
                        ay=0,
                        font=dict(size=10, color=color_mapping_dict.get(variable, "black")),
                        arrowcolor=color_mapping_dict.get(variable, "black"),
                        arrowwidth=1.0
                    )
                )
        tickvals = pd.date_range(
            start=pd.Timestamp(year=data.index.min().year, month=1, day=1),
            end=data.index.max(),
            freq=f"{num_years_interval_x_axis}YS"
            ).to_pydatetime()
        # Update layout with dual y-axes
        fig.update_layout(
            title={
                "text": title,
                "x": 0.5,  # Center the title
                "xanchor": "center",  # Anchor the title at the center
                "yanchor": "top",  # Anchor the title at the top
                "y": 0.95  # Adjust the vertical position of the title
            },
            xaxis=dict(
                title=x_axis_title,
                tickformat='%Y',
                gridcolor="lightgrey",  # Set x-axis grid lines to light grey
                tickvals=tickvals
                ),
            yaxis=dict(
                title=y_axis_title,
                gridcolor="lightgrey"  # Set y-axis grid lines to light grey
                ),
            yaxis2=dict(
                title=secondary_y_axis_title,
                overlaying="y",  # Overlay on the primary y-axis
                side="right",  # Place on the right side
                gridcolor="lightgrey",
                ),
            font=self.font,
            margin=margins,
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.paper_bgcolor,
            annotations=annotations,  # Add the annotations
            showlegend=showlegend
        )
        if save_fig:
            if file_name is None or file_path_save is None:
                raise ValueError(f"file_name and file_path must be provided when save_fig is True, is currently as: {file_name}, {file_path_save}")
            else:
                file_name = file_name if file_name.endswith(".pdf") else f"{file_name}.pdf"
                logging.info(f"Saving figure as PDF to: {file_path_save} with name: {file_name}")
                self._save_figure_as_pdf(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save,
                    width=width,
                    height=height,
                    scale=scale
                )
                file_name = file_name.replace(".pdf", ".html") if file_name.endswith(".pdf") else f"{file_name}.html"
                logging.info(f"Saving figure as HTML to: {file_path_save} with name: {file_name}")
                self._save_figure_as_html(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save
                )
        return fig


    def get_fig_crisis_periods_highlighted(
        self,
        data: pd.DataFrame,
        variables: List[str],
        secondary_y_variables: List[str],
        title: str,
        secondary_y_axis_title: str,
        x_axis_title: str,
        y_axis_title: str,
        color_mapping_dict: Dict[str, str],
        crisis_periods_dict: Dict[str, Dict[str, str]]=None,
        num_years_interval_x_axis: int=5,
        recession_shading_color: str="rgba(200,200,200,0.3)",  # Light gray for recessions
        margin_dict: Dict[str, float]=None,
        showlegend: bool=False,
        save_fig: bool=False,
        file_name: str=None,
        file_path: str=None,
        width: int = 800,
        height: int = 600,
        scale: int = 2
        ) -> go.Figure:
        if margin_dict is not None:
            margins = margin_dict
        else:
            margins = self.margin
        if file_path is not None:
            file_path_save = file_path
        elif self.file_path:
            file_path_save = self.file_path
        else:
            file_path_save = r"/"

        # First infer what the start and end date for the plot should be
        start_date = data.index.min()
        end_date = data.index.max()

        fig = go.Figure()
        # Plot variables on the primary y-axis
        for i, var in enumerate(variables):
            if secondary_y_variables and var in secondary_y_variables:
                continue
            fig.add_trace(
                go.Scatter(
                    x=data.index.to_pydatetime(),  # <-- FIXED
                    y=data[var],
                    mode='lines',
                    name=str(var),
                    opacity=1.0,
                    showlegend=showlegend,
                    line=dict(
                        color=color_mapping_dict.get(var, "black"),
                        width=2.0
                    )
                )
            )
        # Plot variables on the secondary y-axis
        if secondary_y_variables:
            for i, var in enumerate(secondary_y_variables):
                if var not in data.columns:
                    logging.info(f"Variable '{var}' not found in data columns. Skipping.")
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=data.index.to_pydatetime(),  # <-- FIXED
                        y=data[var],
                        mode='lines',
                        name=f"{var} (r.h.)",
                        opacity=1.0,
                        showlegend=showlegend,
                        line=dict(
                            color=color_mapping_dict.get(var, "black"),
                            width=2.0
                        ),
                        yaxis="y2"
                    )
                )
        # Add shaded regions as traces (these will appear in the legend)
        if crisis_periods_dict is not None:
            for name, period in crisis_periods_dict.items():
                start = pd.to_datetime(period["start"])
                end = pd.to_datetime(period["end"])
                if start < start_date or end > end_date:
                    continue  # Skip periods outside the data range
                # Clip the start and end to the data range
                fig.add_trace(
                    go.Scatter(
                        x=[start, end, end, start],
                        # y=[data.min().min(), data.min().min(), data.max().max(), data.max().max()],
                        y=[data[variables].min().min(), data[variables].min().min(), data[variables].max().max(), data[variables].max().max()],
                        fill="toself",
                        fillcolor="rgba(255,0,0,0.2)" if not "recession" in name.lower() else recession_shading_color,
                        line=dict(color="rgba(255,0,0,0.2)" if not "recession" in name.lower() else recession_shading_color),
                        name=name,
                        showlegend=showlegend,
                        hoverinfo="skip"
                    )
                )
        tickvals = pd.date_range(
            start=pd.Timestamp(year=data.index.min().year, month=1, day=1),
            end=data.index.max(),
            freq=f"{num_years_interval_x_axis}YS"
            ).to_pydatetime()
        # Add annotations for each variable
        annotations = []
        for variable in data.columns:
            product_data = data[variable]
            if not product_data.empty:
                # Get the last data point for the variable
                x_last = (pd.Timestamp(product_data.index[-1]) + pd.DateOffset(days=12)).to_pydatetime()  # <-- FIXED
                y_last = product_data[-1]# * 3.0
                annotations.append(
                    dict(
                        x=x_last,
                        y=y_last,
                        xref="x",
                        yref="y" if variable not in secondary_y_variables else "y2",
                        text=variable,
                        showarrow=True,
                        arrowhead=1,
                        ax=40,
                        ay=-40,
                        font=dict(size=10, color=color_mapping_dict.get(variable, "black")),
                        arrowcolor=color_mapping_dict.get(variable, "black"),
                        arrowwidth=1.0
                    )
                )
        # Update layout with dual y-axes
        fig.update_layout(
            title={
                "text": title,
                "x": 0.5,  # Center the title
                "xanchor": "center",  # Anchor the title at the center
                "yanchor": "top",  # Anchor the title at the top
                "y": 0.95  # Adjust the vertical position of the title
            },
            xaxis=dict(
                title=x_axis_title,
                tickformat='%Y',
                gridcolor="lightgrey",  # Set x-axis grid lines to light grey
                tickvals=tickvals
                ),
            yaxis=dict(
                title=y_axis_title,
                gridcolor="lightgrey"  # Set y-axis grid lines to light grey
                ),
            yaxis2=dict(
                title=secondary_y_axis_title,
                overlaying="y",  # Overlay on the primary y-axis
                side="right",  # Place on the right side
                gridcolor="lightgrey",
                ),
            font=self.font,
            margin=margins,
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.paper_bgcolor,
            annotations=annotations,  # Add the annotations
            showlegend=showlegend
        )
        if save_fig:
            if file_name is None or file_path_save is None:
                raise ValueError(f"file_name and file_path must be provided when save_fig is True, is currently as: {file_name}, {file_path_save}")
            else:
                file_name = file_name if file_name.endswith(".pdf") else f"{file_name}.pdf"
                logging.info(f"Saving figure as PDF to: {file_path_save} with name: {file_name}")
                self._save_figure_as_pdf(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save,
                    width=width,
                    height=height,
                    scale=scale
                )
                file_name = file_name.replace(".pdf", ".html") if file_name.endswith(".pdf") else f"{file_name}.html"
                logging.info(f"Saving figure as HTML to: {file_path_save} with name: {file_name}")
                self._save_figure_as_html(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save
                )
        return fig


    def get_model_comparison_bar_plot(
        self,
        data: pd.DataFrame,
        evaluation_score_col_name: str,
        title: str,
        x_axis_title: str,
        y_axis_title: str,
        color_mapping_dict: Dict[str, str],
        margin_dict: Dict[str, float]=None,
        display_red_line_at_y_equals_1: bool=True,
        textfont_size: int=16,
        showlegend: bool=False,
        save_fig: bool=False,
        file_name: str=None,
        file_path: str=None,
        width: int = 800,
        height: int = 600,
        scale: int = 2
        ) -> go.Figure:
        if margin_dict is not None:
            margins = margin_dict
        else:
            margins = self.margin
        if file_path is not None:
            file_path_save = file_path
        elif self.file_path:
            file_path_save = self.file_path
        else:
            file_path_save = r"/"

        # Prepare x-tick labels and sort by feature count
        data["feature_count"] = data["feature_names_in"].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else len(x))
        data = data.sort_values("feature_count")
        x_labels = data["feature_names_in"].unique().astype(str).tolist()
        group_boundaries = [i for i in range(1, len(x_labels)) if x_labels[i] != x_labels[i-1]]

        unique_models = list(data["model_type"].unique())
        n_models = len(unique_models)
        bar_width = 0.8 / n_models  # total bar width (default 0.8)

        fig = go.Figure()
        for idx, model in enumerate(unique_models):
            model_data = data[data["model_type"] == model]
            fig.add_trace(go.Bar(
                x=model_data["feature_names_in"].astype(str),
                y=model_data[evaluation_score_col_name],
                name=model,
                text=model_data[evaluation_score_col_name].round(3),
                textposition="outside",
                textfont=dict(size=textfont_size),
                showlegend=showlegend,
                marker_color=color_mapping_dict.get(model, "grey"),
                offsetgroup=model
            ))

        # Add vertical dashed lines at group boundaries
        for boundary in group_boundaries:
            fig.add_shape(
                type="line",
                x0=boundary-0.5, x1=boundary-0.5,  # Place between bars
                y0=0, y1=1.1,
                line=dict(color="grey", dash="dash"),
                xref="x", yref="y"
            )

        if display_red_line_at_y_equals_1:
            # Add horizontal red line at y=1
            fig.add_shape(
                type="line",
                x0=-0.5, x1=len(x_labels)-0.5,
                y0=1, y1=1,
                line=dict(color="red", width=2),
                xref="x", yref="y"
            )

        # Use offset calculation to place annotations correctly
        for x_idx, x_val in enumerate(x_labels):
            x_group_data = data[data["feature_names_in"].astype(str) == x_val]
            for m_idx, model in enumerate(unique_models):
                bar_data = x_group_data[x_group_data["model_type"] == model]
                if not bar_data.empty:
                    offset = (-0.4 + bar_width/2) + m_idx * bar_width  # default centers first bar at left of group
                    # Place annotation under bar (model type)
                    fig.add_annotation(
                        x=x_idx + offset,
                        y=0,
                        text=model,
                        showarrow=False,
                        yshift=-120,
                        textangle=90,
                        font=dict(size=textfont_size),
                        xref="x",
                        yref="y"
                    )
        fig.update_layout(
            title={
                "text": title,
                "x": 0.5,  # Center the title
                "xanchor": "center",  # Anchor the title at the center
                "yanchor": "top",  # Anchor the title at the top
                "y": 0.95  # Adjust the vertical position of the title
            },
            xaxis=dict(
                title=x_axis_title,
                #gridcolor="lightgrey",  # Set x-axis grid lines to light grey
                ),
            yaxis=dict(
                title=y_axis_title,
                gridcolor="lightgrey"  # Set y-axis grid lines to light grey
                ),
            barmode="group",
            font=self.font,
            margin=margins,
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.paper_bgcolor,
            showlegend=showlegend,
            uniformtext_minsize=textfont_size,
            uniformtext_mode="show",
        )
        if save_fig:
            if file_name is None or file_path_save is None:
                raise ValueError(f"file_name and file_path must be provided when save_fig is True, is currently as: {file_name}, {file_path_save}")
            else:
                file_name = file_name if file_name.endswith(".pdf") else f"{file_name}.pdf"
                logging.info(f"Saving figure as PDF to: {file_path_save} with name: {file_name}")
                self._save_figure_as_pdf(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save,
                    width=width,
                    height=height,
                    scale=scale
                )
                file_name = file_name.replace(".pdf", ".html") if file_name.endswith(".pdf") else f"{file_name}.html"
                logging.info(f"Saving figure as HTML to: {file_path_save} with name: {file_name}")
                self._save_figure_as_html(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save
                )
        return fig


    def get_model_comparison_regime_counts_bar_plot(
        self,
        data: pd.DataFrame,
        evaluation_score_col_name: str,
        title: str,
        x_axis_title: str,
        y_axis_title: str,
        color_mapping_dict: Dict[str, str],
        show_score: bool=True,
        counts_display: str="absolute",
        text_color_class_0: str="white",
        text_color_class_1: str="white",
        color_class_0: str="lightblue",
        color_class_1: str="darkblue",
        margin_dict: Dict[str, float]=None,
        textfont_size: int=16,
        showlegend: bool=False,
        save_fig: bool=False,
        file_name: str=None,
        file_path: str=None,
        width: int = 800,
        height: int = 600,
        scale: int = 2
        ) -> go.Figure:
        if margin_dict is not None:
            margins = margin_dict
        else:
            margins = self.margin
        if file_path is not None:
            file_path_save = file_path
        elif self.file_path:
            file_path_save = self.file_path
        else:
            file_path_save = r"/"

        # Ensure feature count and sort
        data["feature_count"] = data["feature_names_in"].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else len(x))
        data = data.sort_values("feature_count")

        # detect count columns (accept int 0/1 or string '0'/'1')
        col0 = 0 if 0 in data.columns else ('0' if '0' in data.columns else None)
        col1 = 1 if 1 in data.columns else ('1' if '1' in data.columns else None)

        if col0 is None or col1 is None:
            raise ValueError("Expected count columns named 0 and 1 (int or '0'/'1' strings) in the dataframe for stacked counts.")

        # Category labels and models
        x_labels = data["feature_names_in"].astype(str).unique().tolist()
        unique_models = list(data["model_type"].unique())
        n_models = len(unique_models)
        bar_group_total_width = 0.8
        bar_width = bar_group_total_width / max(1, n_models)

        # aggregate counts per (feature_names_in, model_type)
        grouped = data.groupby(["feature_names_in", "model_type"]).agg({col0: "sum", col1: "sum", evaluation_score_col_name: "first"}).reset_index()

        # numeric center positions for categories
        x_idx_map = {label: idx for idx, label in enumerate(x_labels)}

        fig = go.Figure()

        # Build stacked bars per model (class 0 bottom, class 1 top)
        for m_idx, model in enumerate(unique_models):
            offset = (-bar_group_total_width / 2) + (bar_width / 2) + m_idx * bar_width
            x_positions = []
            y0_values = []
            y1_values = []
            eval_texts = []
            text0 = []
            text1 = []

            for xl in x_labels:
                row = grouped[(grouped["feature_names_in"].astype(str) == str(xl)) & (grouped["model_type"] == model)]
                if not row.empty:
                    y0 = float(row.iloc[0][col0])
                    y1 = float(row.iloc[0][col1])
                    score = row.iloc[0][evaluation_score_col_name]
                else:
                    y0 = 0.0
                    y1 = 0.0
                    score = None

                total = y0 + y1
                if counts_display == "relative":
                    # avoid division by zero
                    y0_disp = (y0 / total) if total > 0 else 0.0
                    y1_disp = (y1 / total) if total > 0 else 0.0
                    text0.append(f"{round(y0_disp * 100, 1)}%")
                    text1.append(f"{round(y1_disp * 100, 1)}%")
                else:
                    y0_disp = y0
                    y1_disp = y1
                    text0.append(f"{int(y0)}")
                    text1.append(f"{int(y1)}")

                x_positions.append(x_idx_map[xl] + offset)
                y0_values.append(y0_disp)
                y1_values.append(y1_disp)
                eval_texts.append(score)

            model_color = color_mapping_dict.get(model, "grey")

            # bottom (class 0)
            fig.add_trace(go.Bar(
                x=x_positions,
                y=y0_values,
                name=f"{model} - 0",
                marker_color=color_class_0,
                opacity=0.6,
                offsetgroup=model,
                legendgroup=f"{model}_0",
                showlegend=showlegend,
                width=bar_width,
                text=text0,
                textposition="inside",
                textfont=dict(size=textfont_size, color=text_color_class_0)
            ))
            # top (class 1)
            fig.add_trace(go.Bar(
                x=x_positions,
                y=y1_values,
                name=f"{model} - 1",
                marker_color=color_class_1,
                opacity=1.0,
                offsetgroup=model,
                legendgroup=f"{model}_1",
                showlegend=showlegend,
                width=bar_width,
                text=text1,
                textposition="inside",
                textfont=dict(size=textfont_size, color=text_color_class_1)
            ))

            # evaluation score annotation above each stacked bar (always on displayed scale)
            if show_score:
                for xi, y0v, y1v, score in zip(x_positions, y0_values, y1_values, eval_texts):
                    total_h = y0v + y1v
                    if score is not None:
                        fig.add_annotation(
                            x=xi,
                            y=total_h,
                            text=str(round(float(score), 3)),
                            showarrow=False,
                            yanchor="bottom",
                            font=dict(size=max(10, textfont_size-2)),
                            xref="x",
                            yref="y"
                        )

        # Add model-type annotations under each individual bar (replicates original snippet)
        for x_idx, x_val in enumerate(x_labels):
            x_group_data = data[data["feature_names_in"].astype(str) == x_val]
            for m_idx, model in enumerate(unique_models):
                bar_data = x_group_data[x_group_data["model_type"] == model]
                if not bar_data.empty:
                    offset = (-bar_group_total_width / 2) + (bar_width / 2) + m_idx * bar_width
                    fig.add_annotation(
                        x=x_idx + offset,
                        y=0,
                        text=model,
                        showarrow=False,
                        yshift=-120,
                        textangle=90,
                        font=dict(size=textfont_size),
                        xref="x",
                        yref="y"
                    )

        # set x ticks to category centers
        tickvals = list(range(len(x_labels)))
        fig.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=x_labels,
            title=x_axis_title
        )

        fig.update_layout(
            title={
                "text": title,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "y": 0.95
            },
            xaxis=dict(),
            yaxis=dict(
                title=y_axis_title,
                gridcolor="lightgrey"
            ),
            barmode="stack",
            font=self.font,
            margin=margins,
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.paper_bgcolor,
            showlegend=showlegend,
            uniformtext_minsize=textfont_size,
            uniformtext_mode="show",
        )
        if save_fig:
            if file_name is None or file_path_save is None:
                raise ValueError(f"file_name and file_path must be provided when save_fig is True, is currently as: {file_name}, {file_path_save}")
            else:
                file_name = file_name if file_name.endswith(".pdf") else f"{file_name}.pdf"
                logging.info(f"Saving figure as PDF to: {file_path_save} with name: {file_name}")
                self._save_figure_as_pdf(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save,
                    width=width,
                    height=height,
                    scale=scale
                )
                file_name = file_name.replace(".pdf", ".html") if file_name.endswith(".pdf") else f"{file_name}.html"
                logging.info(f"Saving figure as HTML to: {file_path_save} with name: {file_name}")
                self._save_figure_as_html(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save
                )
        return fig


    def get_fig_histogram(
        self,
        data: pd.DataFrame,
        variables: List[str],
        title: str="",
        x_axis_title: str="",
        y_axis_title: str="",
        color_mapping_dict: Dict[str, str]=None,
        color_discrete_sequence: List[str]=None,
        margin_dict: Dict[str, float]=None,
        save_fig: bool=False,
        file_name: str=None,
        file_path: str=None,
        showlegend: bool=True,
        opacity: float=1.0,
        width: int = 800,
        height: int = 600,
        scale: int = 2,
        histnorm: str="probability density",
        draw_vertical_line_at_0: bool=True,
        draw_normal_distribution: bool=True
        ) -> go.Figure:
        if margin_dict is not None:
                margins = margin_dict
        else:
            margins = self.margin
        if file_path is not None:
            file_path_save = file_path
        elif self.file_path:
            file_path_save = self.file_path
        else:
            file_path_save = r"/"

        if data.empty:
            raise Exception(f"Provided empty pd.DataFrame: {data}")
        # Reduce the data to the selction
        data = data[variables]

        # Set the default color sequence as variable to use
        if color_discrete_sequence is None:
            color_discrete_sequence = self.color_discrete_sequence_default

        # create a big plot with the amout of subplots
        hist_fig = make_subplots(
            rows=1,
            cols=len(data.columns),
            shared_yaxes=False,
            subplot_titles=variables,
            )
        for i, var in enumerate(variables):
            histogram = go.Histogram(
                x=data[var].values,
                name=str(var),
                marker_color=color_mapping_dict.get(var, color_discrete_sequence[i % len(color_discrete_sequence)]),
                histnorm=histnorm
                )
            hist_fig.add_trace(
                histogram,
                    row=1,
                    col=i+1
                    )
            if draw_normal_distribution:
                # Add the normal distribution to plot against it
                x_vals = np.linspace(
                        start=min(data[var].values),
                        stop=max(data[var].values),
                        num=10000
                        )
                norm_pdf = stats.norm.pdf(
                        x=x_vals,
                        loc=np.mean(data[var].values),
                        scale=np.std(data[var].values)
                        )
                hist_fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=norm_pdf,
                        name=f"{var}_norm",
                        marker_color="#9b0a7d",
                        mode="lines",
                        showlegend=showlegend,
                        opacity=opacity
                        ),
                        row=1,
                        col=i+1
                    )
            # Add a vertical solid line at the x-value: 0
            counts, bin_edges = np.histogram(data[var].values, bins="auto", density=True)
            y_max = max(counts)  # Maximum value of the normal distribution curve

            # Add a vertical solid line at x=0 spanning the inferred y-axis height
            if draw_vertical_line_at_0:
                hist_fig.add_trace(
                    go.Scatter(
                        x=[0, 0],  # Vertical line at x=0
                        y=[0, y_max],  # Extend from y=0 to max_y
                        name=f"{var}_vertical",
                        marker_color="red",
                        mode="lines",
                        showlegend=showlegend
                        ),
                    row=1,
                    col=i + 1
                )
        for i, var in enumerate(variables):
            hist_fig.update_xaxes(
                title_text=x_axis_title if isinstance(x_axis_title, str) else x_axis_title[i],
                gridcolor="lightgrey",
                row=1,
                col=i+1,
                title_standoff=10,
                title_font=dict(size=16),
                title=dict(text=x_axis_title, standoff=10)
            )
            hist_fig.update_yaxes(
                title_text=y_axis_title if isinstance(y_axis_title, str) else y_axis_title[i],
                gridcolor="lightgrey",
                row=1,
                col=i+1,
                title_standoff=10,
                title_font=dict(size=16),
                title=dict(text=y_axis_title, standoff=10)
            )
        hist_fig.update_layout(
            title={
                "text": title,
                "x": 0.5,  # Center the title
                "xanchor": "center",  # Anchor the title at the center
                "yanchor": "top",  # Anchor the title at the top
                "y": 0.95  # Adjust the vertical position of the title
            },
            xaxis=dict(
                # title=x_axis_title,
                gridcolor="lightgrey",  # Set x-axis grid lines to light grey
                ),
            yaxis=dict(
                # title=y_axis_title,
                gridcolor="lightgrey"  # Set y-axis grid lines to light grey
                ),
            font=self.font,
            margin=margins,
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.paper_bgcolor,
            # annotations=annotations,  # Add the annotations
            showlegend=showlegend
        )
        if save_fig:
            if file_name is None or file_path_save is None:
                raise ValueError(f"file_name and file_path must be provided when save_fig is True, is currently as: {file_name}, {file_path_save}")
            else:
                file_name = file_name if file_name.endswith(".pdf") else f"{file_name}.pdf"
                logging.info(f"Saving figure as PDF to: {file_path_save} with name: {file_name}")
                self._save_figure_as_pdf(
                    fig=hist_fig,
                    file_name=file_name,
                    file_path=file_path_save,
                    width=width,
                    height=height,
                    scale=scale
                )
                file_name = file_name.replace(".pdf", ".html") if file_name.endswith(".pdf") else f"{file_name}.html"
                logging.info(f"Saving figure as HTML to: {file_path_save} with name: {file_name}")
                self._save_figure_as_html(
                    fig=hist_fig,
                    file_name=file_name,
                    file_path=file_path_save
                )
        return hist_fig


    def plot_granger_test_results(
        self,
        data: pd.DataFrame,
        variables: List[str],
        secondary_y_variables: List[str],
        title: str,
        secondary_y_axis_title: str,
        color_discrete_sequence: List[str],
        x_axis_title: str,
        y_axis_title: str,
        color_mapping_dict: Dict[str, str],
        significance_level: float=0.05,
        margin_dict: Dict[str, float]=None,
        showlegend: bool=False,
        save_fig: bool=False,
        file_name: str=None,
        file_path: str=None,
        width: int = 800,
        height: int = 600,
        scale: int = 2
        ) -> go.Figure:
        if margin_dict is not None:
            margins = margin_dict
        else:
            margins = self.margin
        if file_path is not None:
            file_path_save = file_path
        elif self.file_path:
            file_path_save = self.file_path
        else:
            file_path_save = r"/"

        fig = go.Figure()
        # Plot each Metric as a separate line on the primary Y-axis
        for i, var in enumerate(variables):
            metric_data = data[data['Metric'] == var]
            fig.add_trace(
                go.Scatter(
                    x=metric_data['Lag'],
                    y=metric_data["Test-Statistic"],
                    mode='lines+markers',
                    name=f"{var}",
                    yaxis="y1",  # Assign to primary Y-axis
                    showlegend=showlegend,
                    line=dict(
                                color=color_mapping_dict.get(var, color_discrete_sequence[i % len(color_discrete_sequence)]),
                                width=2.0
                            ),
                )
            )
        # Plot variables on the secondary y-axis
        if secondary_y_variables:
            for i, var in enumerate(secondary_y_variables):
                if var not in data.columns:
                    logging.info(f"Variable '{var}' not found in data columns. Skipping.")
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=data["Lag"],  # <-- FIXED
                        y=data[var],
                        mode='lines',
                        name=f"{var} (r.h.)",
                        opacity=1.0,
                        showlegend=showlegend,
                        line=dict(
                            color=color_mapping_dict.get(var, color_discrete_sequence[i % len(color_discrete_sequence)]),
                            width=2.0
                        ),
                        yaxis="y2"
                    )
                )
        # Add a horizontal line for the significance level on the secondary Y-axis
        fig.add_trace(
            go.Scatter(
                x=[data['Lag'].min(), data['Lag'].max()],
                y=[significance_level, significance_level],
                mode='lines',
                name=f"Significance Level ({significance_level}) (r.h.)",
                line=dict(color="green", dash="dot"),
                yaxis="y2",  # Assign to secondary Y-axis
                showlegend=showlegend,
            )
        )
        color_mapping_dict[f"Significance Level ({significance_level}) (r.h.)"] = "green"
        # Add annotations for each variable
        annotations = []
        y_last_list = []
        min_distance = 0.3  # Minimum vertical distance between labels

        for i, var in enumerate(variables.tolist() + secondary_y_variables + [f"Significance Level ({significance_level}) (r.h.)"]):
            if var in secondary_y_variables:
                metric_data = data[["Lag", var]]
                y_last = metric_data[var].iloc[-1]
            elif var == f"Significance Level ({significance_level}) (r.h.)":
                y_last = significance_level
            else:
                metric_data = data[data['Metric'] == var]
                if not metric_data.empty and not metric_data["Test-Statistic"].empty:
                    y_last = metric_data["Test-Statistic"].iloc[-1]
                else:
                    continue

            # Check for overlap and add offset if needed
            offset = 0
            for prev_y in y_last_list:
                if abs(y_last + offset - prev_y) < min_distance:
                    offset += min_distance
            y_last_list.append(y_last + offset)
            if var == f"Significance Level ({significance_level}) (r.h.)":
                x_last = data["Lag"].max()
            else:
                x_last = metric_data["Lag"].max()
            annotations.append(
                dict(
                    x=x_last,
                    y=y_last,
                    xref="x",
                    yref="y" if var not in secondary_y_variables else "y2",
                    text=var if var not in secondary_y_variables else f"{var} (r.h.)",
                    showarrow=True,
                    arrowhead=1,
                    ax=30,
                    ay=-30 if offset != 0 else +10,
                    font=dict(size=10, color=color_mapping_dict.get(var, "black")),
                    arrowcolor=color_mapping_dict.get(var, "black"),
                    arrowwidth=1.0
                )
            )
        # Update layout to include a secondary Y-axis
        fig.update_layout(
            title={
                "text": title,
                "x": 0.5,  # Center the title
                "xanchor": "center",  # Anchor the title at the center
                "yanchor": "top",  # Anchor the title at the top
                "y": 0.95  # Adjust the vertical position of the title
            },
            xaxis=dict(
                title=x_axis_title,
                gridcolor="lightgrey",  # Set x-axis grid lines to light grey
                ),
            yaxis=dict(
                title=y_axis_title,
                gridcolor="lightgrey"  # Set y-axis grid lines to light grey
                ),
            yaxis2=dict(
                title=secondary_y_axis_title,
                overlaying="y",  # Overlay on the primary y-axis
                side="right",  # Place on the right side
                gridcolor="lightgrey",
                ),
            font=self.font,
            margin=margins,
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.paper_bgcolor,
            annotations=annotations,  # Add the annotations
            showlegend=showlegend
        )
        if save_fig:
            if file_name is None or file_path_save is None:
                raise ValueError(f"file_name and file_path must be provided when save_fig is True, is currently as: {file_name}, {file_path_save}")
            else:
                file_name = file_name if file_name.endswith(".pdf") else f"{file_name}.pdf"
                logging.info(f"Saving figure as PDF to: {file_path_save} with name: {file_name}")
                self._save_figure_as_pdf(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save,
                    width=width,
                    height=height,
                    scale=scale
                )
                file_name = file_name.replace(".pdf", ".html") if file_name.endswith(".pdf") else f"{file_name}.html"
                logging.info(f"Saving figure as HTML to: {file_path_save} with name: {file_name}")
                self._save_figure_as_html(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save
                )
        return fig


    def get_fig_acf(
        self,
        data: pd.DataFrame,
        variables: List[str],
        title: str,
        x_axis_title: str,
        y_axis_title: str,
        bar_color: str="black",
        nlags: int = 40,
        titles: list = None,
        bar_width: float = 0.1,
        dot_size: int = 8,
        dot_color: str = "red",
        confidence_fill_color: str = "rgba(200,200,200,0.3)",
        confidence_line_color: str = 'rgba(255,255,255,0)',
        margin_dict: Dict[str, float]=None,
        showlegend: bool=False,
        save_fig: bool=False,
        file_name: str=None,
        file_path: str=None,
        width: int = 800,
        height: int = 600,
        scale: int = 2
        ) -> go.Figure:
        if margin_dict is not None:
            margins = margin_dict
        else:
            margins = self.margin
        if file_path is not None:
            file_path_save = file_path
        elif self.file_path:
            file_path_save = self.file_path
        else:
            file_path_save = r"/"

        n_plots = len(variables)
        fig = make_subplots(rows=1, cols=n_plots, subplot_titles=titles or variables)
        for i, variable in enumerate(variables):
            acf_vals, confint = acf(data[variable], nlags=nlags, alpha=0.05)
            lags = list(range(len(acf_vals)))[1:]
            acf_vals = acf_vals[1:]
            confint = confint[1:]
            # Slim bars
            fig.add_trace(go.Bar(
                x=lags,
                y=acf_vals,
                name=f"ACF {variable}",
                width=bar_width,
                marker_color=bar_color,
                showlegend=showlegend
            ), row=1, col=i+1)
            # Dots at top
            fig.add_trace(go.Scatter(
                x=lags,
                y=acf_vals,
                mode="markers",
                marker=dict(color=dot_color, size=dot_size),
                name=f"ACF Top {variable}",
                showlegend=showlegend
            ), row=1, col=i+1)
            # Confidence interval
            offset = 0.2
            lags_extended = [lags[0] - offset] + lags + [lags[-1] + offset]
            confint_lower = [confint[0, 0]] + list(confint[:, 0]) + [confint[-1, 0]]
            confint_upper = [confint[0, 1]] + list(confint[:, 1]) + [confint[-1, 1]]
            fig.add_trace(go.Scatter(
                x=lags_extended + lags_extended[::-1],
                y=confint_lower + confint_upper[::-1],
                fill='toself',
                fillcolor=confidence_fill_color,
                line=dict(color=confidence_line_color),
                hoverinfo="skip",
                showlegend=showlegend,
                name=f"95% CI {variable}"
            ), row=1, col=i+1)
            # Set x-axis ticks for each subplot
            fig.update_xaxes(title_text=x_axis_title, tickmode='linear', tick0=1, dtick=1, range=[0.5, nlags+0.5], row=1, col=i+1)
            fig.update_yaxes(title_text=y_axis_title, row=1, col=i+1)

        fig.update_layout(
            bargap=0.1,
            title={
                "text": title,
                "x": 0.5,  # Center the title
                "xanchor": "center",  # Anchor the title at the center
                "yanchor": "top",  # Anchor the title at the top
                "y": 0.95  # Adjust the vertical position of the title
            },
            # xaxis=dict(
            #     title=x_axis_title,
            #     gridcolor="lightgrey",  # Set x-axis grid lines to light grey
            #     ),
            # yaxis=dict(
            #     title=y_axis_title,
            #     gridcolor="lightgrey"  # Set y-axis grid lines to light grey
            #     ),
            font=self.font,
            margin=margins,
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.paper_bgcolor,
            # annotations=annotations,  # Add the annotations
            showlegend=showlegend
        )
        if save_fig:
            if file_name is None or file_path_save is None:
                raise ValueError(f"file_name and file_path must be provided when save_fig is True, is currently as: {file_name}, {file_path_save}")
            else:
                file_name = file_name if file_name.endswith(".pdf") else f"{file_name}.pdf"
                logging.info(f"Saving figure as PDF to: {file_path_save} with name: {file_name}")
                self._save_figure_as_pdf(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save,
                    width=width,
                    height=height,
                    scale=scale
                )
                file_name = file_name.replace(".pdf", ".html") if file_name.endswith(".pdf") else f"{file_name}.html"
                logging.info(f"Saving figure as HTML to: {file_path_save} with name: {file_name}")
                self._save_figure_as_html(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save
                )
        return fig


    def get_fig_pacf(
        self,
        data: pd.DataFrame,
        variables: List[str],
        title: str,
        x_axis_title: str,
        y_axis_title: str,
        bar_color: str="black",
        nlags: int = 40,
        titles: list = None,
        bar_width: float = 0.1,
        dot_size: int = 8,
        dot_color: str = "red",
        confidence_fill_color: str = "rgba(200,200,200,0.3)",
        confidence_line_color: str = 'rgba(255,255,255,0)',
        margin_dict: Dict[str, float]=None,
        showlegend: bool=False,
        save_fig: bool=False,
        file_name: str=None,
        file_path: str=None,
        width: int = 800,
        height: int = 600,
        scale: int = 2
        ) -> go.Figure:
        if margin_dict is not None:
            margins = margin_dict
        else:
            margins = self.margin
        if file_path is not None:
            file_path_save = file_path
        elif self.file_path:
            file_path_save = self.file_path
        else:
            file_path_save = r"/"

        n_plots = len(variables)
        fig = make_subplots(rows=1, cols=n_plots, subplot_titles=titles or variables)
        for i, variable in enumerate(variables):
            pacf_vals, confint = pacf(data[variable], nlags=nlags, alpha=0.05)
            lags = list(range(len(pacf_vals)))[1:]
            pacf_vals = pacf_vals[1:]
            confint = confint[1:]
            # Slim bars
            fig.add_trace(go.Bar(
                x=lags,
                y=pacf_vals,
                name=f"PACF {variable}",
                width=bar_width,
                marker_color=bar_color,
                showlegend=showlegend
            ), row=1, col=i+1)
            # Dots at top
            fig.add_trace(go.Scatter(
                x=lags,
                y=pacf_vals,
                mode="markers",
                marker=dict(color=dot_color, size=dot_size),
                name=f"PACF Top {variable}",
                showlegend=showlegend
            ), row=1, col=i+1)
            # Confidence interval (PACF confint is 2D: lower and upper bounds)
            offset = 0.2
            lags_extended = [lags[0] - offset] + lags + [lags[-1] + offset]
            confint_lower = [confint[0, 0]] + list(confint[:, 0]) + [confint[-1, 0]]
            confint_upper = [confint[0, 1]] + list(confint[:, 1]) + [confint[-1, 1]]
            fig.add_trace(go.Scatter(
                x=lags_extended + lags_extended[::-1],
                y=confint_lower + confint_upper[::-1],
                fill='toself',
                fillcolor=confidence_fill_color,
                line=dict(color=confidence_line_color),
                hoverinfo="skip",
                showlegend=showlegend,
                name=f"95% CI {variable}"
            ), row=1, col=i+1)
            fig.update_xaxes(title_text=x_axis_title, tickmode='linear', tick0=1, dtick=1, range=[0.5, nlags+0.5], row=1, col=i+1)
            fig.update_yaxes(title_text=y_axis_title, row=1, col=i+1)

        fig.update_layout(
            bargap=0.1,
            title={
                "text": title,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "y": 0.95
            },
            font=self.font,
            margin=margins,
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.paper_bgcolor,
            # annotations=annotations,  # Add the annotations
            showlegend=showlegend
        )
        if save_fig:
            if file_name is None or file_path_save is None:
                raise ValueError(f"file_name and file_path must be provided when save_fig is True, is currently as: {file_name}, {file_path_save}")
            else:
                file_name = file_name if file_name.endswith(".pdf") else f"{file_name}.pdf"
                logging.info(f"Saving figure as PDF to: {file_path_save} with name: {file_name}")
                self._save_figure_as_pdf(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save,
                    width=width,
                    height=height,
                    scale=scale
                )
                file_name = file_name.replace(".pdf", ".html") if file_name.endswith(".pdf") else f"{file_name}.html"
                logging.info(f"Saving figure as HTML to: {file_path_save} with name: {file_name}")
                self._save_figure_as_html(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save
                )
        return fig


    def plot_coefs_with_ci(
        self,
        data: pd.DataFrame,
        title: str,
        x_axis_title: str="Parameter (index)",
        y_axis_title: str="Coefficient",
        model_col: str="model_file_name",
        coef_col: str="coef",
        ci_lower_col: str="ci_lower",
        ci_upper_col: str="ci_upper",
        t_col: str="t",
        marker_color: str="black",
        param_index_name: str="param",
        group_width: float=0.6,
        marker_size: int=8,
        textfont_size: int=10,
        showlegend: bool=False,
        save_fig: bool=False,
        file_name: str=None,
        file_path: str=None,
        width: int = 800,
        height: int = 600,
        scale: int = 2
        ) -> go.Figure:
        """
        Dot + short CI caps plot.
        - x-axis: unique params (df index or param column)
        - each point: coefficient for a model (horizontal jitter inside param group)
        - short horizontal error-bars (error_y) from ci_lower/ci_upper
        - t-value shown as text next to marker if available
        - model names printed below each respective marker (rotated)
        """
        # if margin_dict is not None:
        #     margins = margin_dict
        # else:
        #     margins = self.margin
        if file_path is not None:
            file_path_save = file_path
        elif self.file_path:
            file_path_save = self.file_path
        else:
            file_path_save = r"/"
        df_plot = data.copy()
        # ensure param is a column
        if df_plot.index.name != param_index_name:
            df_plot = df_plot.reset_index().rename(columns={df_plot.columns[0]: param_index_name}) if param_index_name not in df_plot.columns else df_plot
        if param_index_name not in df_plot.columns:
            df_plot[param_index_name] = df_plot.index.astype(str)

        # map params to integer x positions (params are primary x-axis categories)
        params_unique = list(df_plot[param_index_name].astype(str).unique())
        param_to_x = {p: i for i, p in enumerate(params_unique)}

        # within-each-param ordering to offset multiple models
        df_plot["_within_idx"] = df_plot.groupby(param_index_name).cumcount()
        df_plot["_within_count"] = df_plot.groupby(param_index_name)[coef_col].transform("count")

        # compute numeric x positions with small offsets so multiple models per param don't overlap
        def _compute_x(row):
            base = param_to_x[str(row[param_index_name])]
            n = int(max(1, row["_within_count"]))
            if n == 1:
                return float(base)
            slot = int(row["_within_idx"])
            step = group_width / n
            start = -group_width/2 + step/2
            return float(base + start + slot * step)

        df_plot["_xpos"] = df_plot.apply(_compute_x, axis=1)

        # asymmetric error lengths for plotly error_y
        err_plus = (df_plot[ci_upper_col].astype(float) - df_plot[coef_col].astype(float)).values
        err_minus = (df_plot[coef_col].astype(float) - df_plot[ci_lower_col].astype(float)).values

        # t-value text
        def _t_text(val):
            try:
                v = float(val)
                if np.isfinite(v):
                    return f"t={v:.3f}"
            except Exception:
                pass
            return ""

        texts = df_plot[t_col].apply(_t_text).tolist()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_plot["_xpos"],
            y=df_plot[coef_col].astype(float),
            mode="markers+text",
            marker=dict(size=marker_size, color=marker_color),
            showlegend=False,
            error_y=dict(
                type="data",
                symmetric=False,
                array=err_plus,
                arrayminus=err_minus,
                thickness=1.5,
                width=12
            ),
            hovertemplate=
                "model: %{customdata[0]}<br>param: %{customdata[1]}<br>coef: %{y:.6g}<br>ci: (%{customdata[2]:.6g}, %{customdata[3]:.6g})<extra></extra>",
            customdata=np.stack([
                df_plot[model_col].astype(str).values,
                df_plot[param_index_name].astype(str).values,
                df_plot[ci_lower_col].astype(float).values,
                df_plot[ci_upper_col].astype(float).values
            ], axis=1)
        ))
        y_min = float(np.nanmin(df_plot[[coef_col, ci_lower_col]].astype(float).values)) if not df_plot.empty else 0.0
        y_max = float(np.nanmax(df_plot[[coef_col, ci_upper_col]].astype(float).values)) if not df_plot.empty else 1.0
        y_range = y_max - y_min if (y_max - y_min) != 0 else abs(y_max) if y_max != 0 else 1.0
        offset = y_range * 0.06  # ~6% of axis range; adjust as needed

        # separate text-only trace positioned above each marker
        fig.add_trace(go.Scatter(
            x=df_plot["_xpos"],
            y=(df_plot[coef_col].astype(float) + offset),
            mode="text",
            text=texts,
            textfont=dict(size=textfont_size, color="black"),
            hoverinfo="skip",
            showlegend=False
        ))
        # Set x ticks to param centers and label with param names
        tickvals = list(range(len(params_unique)))
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=params_unique)
        fig.update_xaxes(showgrid=True, gridcolor="lightgrey", gridwidth=1)
        fig.update_yaxes(showgrid=True, gridcolor="lightgrey", gridwidth=1)
        # Add model-name annotations below each marker (rotated)
        paper_annotation_y = -0.12  # 12% below the plotting area; adjust if needed

        # compute a sensible bottom margin (pixels) so rotated model names fit
        # small heuristic: base padding + small factor times number of unique model labels
        bottom_margin = 450

        for _, row in df_plot.iterrows():
            fig.add_annotation(
                x=row["_xpos"],
                y=paper_annotation_y,
                text=str(row[model_col]),
                showarrow=False,
                yanchor="top",
                xanchor="center",
                textangle=90,
                font=dict(size=max(7, textfont_size - 1)),
                xref="x",
                yref="paper"   # place relative to paper (figure) coordinates
            )

        # Update layout with dual y-axes
        fig.update_layout(
            title={
                "text": title,
                "x": 0.5,  # Center the title
                "xanchor": "center",  # Anchor the title at the center
                "yanchor": "top",  # Anchor the title at the top
                "y": 0.95  # Adjust the vertical position of the title
            },
            xaxis=dict(
                title=x_axis_title,
                gridcolor="lightgrey",  # Set x-axis grid lines to light grey
                ),
            yaxis=dict(
                title=y_axis_title,
                gridcolor="lightgrey"  # Set y-axis grid lines to light grey
                ),
            font=self.font,
            margin=dict(l=60, r=20, t=30, b=bottom_margin),
            plot_bgcolor=self.plot_bgcolor,
            paper_bgcolor=self.paper_bgcolor,
            # annotations=annotations,  # Add the annotations
            showlegend=showlegend
        )
        if save_fig:
            if file_name is None or file_path_save is None:
                raise ValueError(f"file_name and file_path must be provided when save_fig is True, is currently as: {file_name}, {file_path_save}")
            else:
                file_name = file_name if file_name.endswith(".pdf") else f"{file_name}.pdf"
                logging.info(f"Saving figure as PDF to: {file_path_save} with name: {file_name}")
                self._save_figure_as_pdf(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save,
                    width=width,
                    height=height,
                    scale=scale
                )
                file_name = file_name.replace(".pdf", ".html") if file_name.endswith(".pdf") else f"{file_name}.html"
                logging.info(f"Saving figure as HTML to: {file_path_save} with name: {file_name}")
                self._save_figure_as_html(
                    fig=fig,
                    file_name=file_name,
                    file_path=file_path_save
                )
        return fig