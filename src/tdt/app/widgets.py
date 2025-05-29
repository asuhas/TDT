import ipywidgets as widgets
import pandas as pd
import datetime


def get_layout(maturity_options,weighting_options):
    label_style = {'description_width': '80px'}
    wide_col_width = '40%'  # First column wider
    narrow_col_width = '28%'  # Second and third columns narrower
    row_width = '95%'
    # === Row 1 ===
    start_date = widgets.DatePicker(
        description='Start:',
        value=pd.to_datetime('2019-01-01')
    )
    #start_date.observe(on_change_handler, names='value')

    end_date = widgets.DatePicker(
        description='End:',
        value=pd.to_datetime('2025-01-01')
    )
    #end_date.observe(on_change_handler, names='value')

    dropdown = widgets.Dropdown(
        options=weighting_options,
        description='Weighting:',
        value=weighting_options[0],
        layout=widgets.Layout(width=wide_col_width),
        style=label_style,
    )
    #dropdown.observe(on_change_handler, names='value')


    # === Row 2 ===
    multi_select = widgets.SelectMultiple(
        options=maturity_options,
        description='Choose Legs of Fly',
        value=(maturity_options[0], maturity_options[1], maturity_options[3]),
        rows=3,
        style=label_style
    )
    #multi_select.observe(on_change_handler, names='value')

    tick_mark = widgets.Checkbox(
        value=True,
        description='Long Belly?'
    )
    #tick_mark.observe(on_change_handler, names='value')

    go_button = widgets.Button(description='Refresh')
    #go_button.on_click(lambda b: on_change_handler({'type': 'click', 'name': 'go_button', 'owner': b}))



    # === Combine rows ===

    row1 = widgets.HBox([start_date, end_date, dropdown],
                        layout=widgets.Layout(width=row_width, justify_content='space-between'))
    row2 = widgets.HBox([multi_select, tick_mark, go_button],
                        layout=widgets.Layout(width=row_width, justify_content='space-between'))
    layout = widgets.VBox([row1, row2])

    #display(layout)
    return start_date,end_date,dropdown,multi_select,tick_mark,go_button,layout
