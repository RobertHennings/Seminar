from typing import Dict, List, Tuple
import matplotlib.colors as mcolors
import numpy as np
import os
import logging
import datetime as dt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.io import to_html

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# import config settings with static global variables
# os.chdir(r"/Users/Robert_Hennings/Uni/Master/Seminar/src/seminar_code/data_graphing")

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
        """
        Save a Plotly figure as a PDF file.

        Parameters:
        - fig: The Plotly figure to save.
        - file_path: The path to save the PDF file.
        - width: The width of the figure in pixels.
        - height: The height of the figure in pixels.
        - scale: The resolution scale (default is 2 for high resolution).
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
        """Lightens the given color by multiplying it with the given factor (0-1)."""
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
        """
        Returns a Plotly-style custom diverging continuous color scale from start to end color.
        
        Parameters:
        - start_hex: Hex color for the low end (e.g., '#ff0000')
        - end_hex: Hex color for the high end (e.g., '#0000ff')
        - steps: Number of gradient steps toward the center for each side
        - lightening_factor: Value < 1 to determine how much each step lightens
        
        Returns:
        - List of [position, color] for use in Plotly
        """

        # center_color = "#ffffff"  # divergence midpoint (white)

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
        x_axis_variable: str,
        y_axis_variable: str,
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
            # Add traces for the primary y-axis variables
            for variable in variables:
                country_data = data.query("Entity == @variable")
                fig.add_trace(
                    go.Scatter(
                        x=country_data[x_axis_variable],
                        y=country_data[y_axis_variable],
                        mode='lines+markers',
                        name=variable,
                        line=dict(color=color_mapping_dict.get(variable, "black"))  # Use the mapped color
                    )
                )

            # Add traces for the secondary y-axis variables
            for variable in secondary_y_variables:
                country_data = data.query("Entity == @variable")
                fig.add_trace(
                    go.Scatter(
                        x=country_data[x_axis_variable],
                        y=country_data[y_axis_variable],
                        mode='lines+markers',
                        name=f"{variable} (r.h.)",
                        yaxis="y2",
                        line=dict(color=color_mapping_dict.get(variable, "black"))  # Use the mapped color
                    )
                )

            # Add annotations for each variable
            annotations = []
            for variable in variables + secondary_y_variables:
                country_data = data.query("Entity == @variable")
                if not country_data.empty:
                    # Get the last data point for the variable
                    x_last = country_data[x_axis_variable].iloc[-1] + 1
                    y_last = country_data[y_axis_variable].iloc[-1]
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
                    gridcolor="lightgrey",  # Set x-axis grid lines to light grey
                    dtick=num_years_interval_x_axis  # Set 5-year intervals (60 months)
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
        margin_dict: Dict[str, float]=dict(
                l=20,  # Left margin
                r=20,  # Right margin
                t=90,  # Top margin
                b=10   # Bottom margin
                ),
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
            fig_consumption_production_combine.add_trace(trace, row=1, col=1, secondary_y=(secondary_y_variable in trace.name))
            # Place label near the last point
            if trace.x.size == 0 or trace.y.size == 0:
                continue
            xval = trace.x[-1] +1
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
            fig_consumption_production_combine.add_trace(trace, row=2, col=1, secondary_y=(secondary_y_variable in trace.name))
            if trace.x.size == 0 or trace.y.size == 0:
                continue
            xval = trace.x[-1] +1
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
        fig_consumption_production_combine.update_xaxes(title_text=x_axis_title, gridcolor="lightgrey", dtick=num_years_interval_x_axis, row=1, col=1)
        fig_consumption_production_combine.update_xaxes(title_text=x_axis_title, gridcolor="lightgrey", dtick=num_years_interval_x_axis, row=2, col=1)

        start_year = fig_production.data[0].x[0]
        end_year = fig_production.data[0].x[-1]
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
                # dtick=num_years_interval_x_axis,
                # type="date",  # <-- Explicitly set axis type
                # tickformat="%Y",  # <-- Optional: show years only
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
        margin_dict: Dict[str, float]=dict(
                l=20,  # Left margin
                r=20,  # Right margin
                t=90,  # Top margin
                b=10   # Bottom margin
                ),
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

        fig.update_xaxes(title_text=x_axis_title, row=1, col=1, gridcolor="lightgrey")
        fig.update_xaxes(title_text=x_axis_title, row=2, col=1, gridcolor="lightgrey")

        fig.update_layout(
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
            #     # dtick=num_years_interval_x_axis
            # ),
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
                # tickformat='%Y-%m',
                gridcolor="lightgrey",  # Set x-axis grid lines to light grey
                # dtick=num_years_interval_x_axis
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
            y=data['HICP'].reindex(data.index),
            name='HICP',
            mode='lines+markers',
            showlegend=False,
            line=dict(color=cpi_color, width=2)
        ))
        # Add annotations for each variable
        annotations = []
        for variable in variables + ["HICP"]:
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
                # tickformat='%Y-%m',
                gridcolor="lightgrey",  # Set x-axis grid lines to light grey
                # dtick=num_years_interval_x_axis
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
        # x_axis_variable: str,
        # y_axis_variable: str,
        x_axis_title: str,
        y_axis_title: str,
        color_mapping_dict: Dict[str, str],
        num_years_interval_x_axis: int=5,
        margin_dict: Dict[str, float]=dict(
                l=20,  # Left margin
                r=20,  # Right margin
                t=50,  # Top margin
                b=10   # Bottom margin
                ),
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
                    print(f"Variable '{var}' not found in data columns. Skipping.")
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
                # tickformat='%Y-%m',
                gridcolor="lightgrey",  # Set x-axis grid lines to light grey
                # dtick=num_years_interval_x_axis
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
        # x_axis_variable: str,
        # y_axis_variable: str,
        x_axis_title: str,
        y_axis_title: str,
        color_mapping_dict: Dict[str, str],
        num_years_interval_x_axis: int=5,
        margin_dict: Dict[str, float]=dict(
                l=20,  # Left margin
                r=20,  # Right margin
                t=50,  # Top margin
                b=10   # Bottom margin
                ),
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
                    print(f"Variable '{var}' not found in data columns. Skipping.")
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
                # tickformat='%Y-%m',
                gridcolor="lightgrey",  # Set x-axis grid lines to light grey
                # dtick=num_years_interval_x_axis
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
        margin_dict: Dict[str, float]=dict(
                l=20,  # Left margin
                r=20,  # Right margin
                t=50,  # Top margin
                b=10   # Bottom margin
                ),
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
                    print(f"Variable '{var}' not found in data columns. Skipping.")
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
                # tickformat='%Y-%m',
                gridcolor="lightgrey",  # Set x-axis grid lines to light grey
                # dtick=num_years_interval_x_axis
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