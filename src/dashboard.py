from PoemGenerator import poem_replacer
import ipywidgets as widgets

html = widgets.HTML(
    value="<H2>Poem Manipulator</H2>Enter your poem in the text area, select options beneath, choose number of substitutions, then click Manipulate!")
log = widgets.Output(layout={'border': '1px solid black', 'height': '40%', 'width': '97%'})
with log:
    print("Log notes will appear here")
header_box = widgets.VBox([html, log])

input_poem = widgets.Textarea(value='Enter poem here', placeholder='Enter poem here', layout={'border': '1px solid black', 'height': '100%', 'width': '95%'})

layout = widgets.Layout(width='auto', height='auto')
substitution_slider = widgets.IntSlider(min=0, max=100, value=10)
substitution_label = widgets.Label('Percent to replace: ', layout=widgets.Layout(width='40%'))
substitution_box = widgets.HBox([substitution_label, substitution_slider])

generate_button =  widgets.Button(description='Manipulate!', disabled=False)

min_wn_slider = widgets.IntSlider(min=-1, max=10, value=0)
min_wn_label = widgets.Label('Min WN distance: ', layout=widgets.Layout(width='40%'))
min_wn_box = widgets.HBox([min_wn_label, min_wn_slider])

max_wn_slider = widgets.IntSlider(min=-1, max=10, value=-1)
max_wn_label = widgets.Label('Max WN distance: ', layout=widgets.Layout(width='40%'))
max_wn_box = widgets.HBox([max_wn_label, max_wn_slider])

pos = widgets.Checkbox(True, layout=widgets.Layout(justify_content="flex-start"))
pos_label = widgets.Label('Use Part of Speech: ', layout=widgets.Layout(width='40%'))
pos_box = widgets.HBox([pos_label, pos])

rhyme = widgets.Checkbox(True)
rhyme_label = widgets.Label('Use Rhyme: ', layout=widgets.Layout(width='40%'))
rhyme_box = widgets.HBox([rhyme_label, rhyme])

anagrams = widgets.Checkbox(False)
anagrams_label = widgets.Label('Use Anagrams: ', layout=widgets.Layout(width='40%'))
anagrams_box = widgets.HBox([anagrams_label, anagrams])

syllables = widgets.Checkbox(True)
syllables_label = widgets.Label('Use Syllables: ', layout=widgets.Layout(width='40%'))
syllables_box = widgets.HBox([syllables_label, syllables])

uoi = widgets.RadioButtons(value='union', options=['union', 'intersection'])
uoi_label = widgets.Label('Combine possibles by: ', layout=widgets.Layout(width='40%'))
uoi_box = widgets.HBox([uoi_label, uoi])

left_box = widgets.VBox([substitution_box, generate_button], layout=widgets.Layout(width='80%', height='auto'))
right_box = widgets.VBox([min_wn_box, max_wn_box, pos_box, syllables_box, rhyme_box, anagrams_box, uoi_box], layout=widgets.Layout(width='80%', height='auto'))
box = widgets.HBox([left_box, right_box])

out = widgets.Output(layout={'border': '1px solid black', 'height': '100%', 'width': '95%'})
with out:
    print("Output poem will appear here")
    
app = widgets.AppLayout(header=header_box, left_sidebar=input_poem, footer=box, center=None, right_sidebar=out)


def process(b):
    PR = poem_replacer('lexicon.json', use_cmu=True, use_pos=pos.value, use_anagrams=anagrams.value, use_syllables=syllables.value, use_rhyme=rhyme.value, union_or_intersection=uoi.value, min_wn_distance=min_wn_slider.value, max_wn_distance=max_wn_slider.value)
    number_to_replace, to_replace, words = PR.process(input_poem.value, substitution_slider.value)
    log.clear_output()
    log.append_stdout(number_to_replace)
    log.append_stdout('tokens to be replaced:' + ', '.join(to_replace))
    out.clear_output()
    out.append_stdout(''.join(words))

generate_button.on_click(process)

app