from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FloatField, SelectField
from wtforms.validators import DataRequired, NumberRange

class MyForm(FlaskForm):
    class Meta:  # Ignoring CSRF security feature.
        csrf = False

    input_field = StringField(label='Input:', id='input_field',
                              validators=[DataRequired()],
                              render_kw={'style': 'width:50%'})

    magnitude = FloatField(
        label='Magnitude:',
        validators=[DataRequired(), NumberRange(min=-3, max=3)],
        render_kw={'style': 'width:20%', 'step': '0.1'}  # Adjust 'step' for desired increment
    )

    text_options = SelectField(label='Select Option:',
        choices= [
    ('./nlp/data/vectors/Confident_Insecure.pt', 'Insecure - Confident'),
    ('./nlp/data/vectors/Evasive_Direct.pt', 'Direct - Evasive'),
    ('./nlp/data/vectors/Diplomatic_Blunt.pt', 'Blunt - Diplomatic'),
    ('./nlp/data/vectors/Inquisitive_Disinterested.pt', 'Disinterested - Inquisitive'),
    ('./nlp/data/vectors/Thoughtful_Impulsive.pt', 'Impulsive - Thoughtful'),
    ('./nlp/data/vectors/Nonchalant_Concerned.pt', 'Concerned - Nonchalant'),
    ('./nlp/data/vectors/Flippant_Serious.pt', 'Serious - Flippant'),
    ('./nlp/data/vectors/Precise_Vague.pt', 'Vague - Precise'),
    ('./nlp/data/vectors/Rambling_Concise.pt', 'Concise - Rambling'),
    ('./nlp/data/vectors/Analytical_Intuitive.pt', 'Intuitive - Analytical'),
    ('./nlp/data/vectors/Assertive_Passive.pt', 'Passive - Assertive'),
    ('./nlp/data/vectors/Considerate_Thoughtless.pt', 'Thoughtless - Considerate'),
    ('./nlp/data/vectors/Elusive_Clear.pt', 'Clear - Elusive'),
    ('./nlp/data/vectors/Candid_Guarded.pt', 'Guarded - Candid'),
    ('./nlp/data/vectors/Defensive_Open.pt', 'Open - Defensive'),
    ('./nlp/data/vectors/Engaging_Detached.pt', 'Detached - Engaging'),
    ('./nlp/data/vectors/Reserved_Outgoing.pt', 'Outgoing - Reserved'),
    ('./nlp/data/vectors/Empathetic_Unsympathetic.pt', 'Unsympathetic - Empathetic'),
    ('./nlp/data/vectors/Concise_Lengthy.pt', 'Lengthy - Concise'),
    ('./nlp/data/vectors/Enthusiastic_Apathetic.pt', 'Apathetic - Enthusiastic'),
    ('./nlp/data/vectors/Introspective_Extrospective.pt', 'Extrospective - Introspective'),
    ('./nlp/data/vectors/Polite_Rude.pt', 'Rude - Polite'),
    ('./nlp/data/vectors/Indecisive_Decisive.pt', 'Decisive - Indecisive'),
    ('./nlp/data/vectors/Dismissive_Receptive.pt', 'Receptive - Dismissive'),
    ('./nlp/data/vectors/Deliberate_Hasty.pt', 'Hasty - Deliberate'),
    ('./nlp/data/vectors/Informative_Misleading.pt', 'Misleading - Informative'),
    ('./nlp/data/vectors/Focused_Distracted.pt', 'Distracted - Focused'),
    ('./nlp/data/vectors/Perplexed_Clear.pt', 'Clear - Perplexed'),
    ('./nlp/data/vectors/Cooperative_Uncooperative.pt', 'Uncooperative - Cooperative'),
    ('./nlp/data/vectors/Inattentive_Attentive.pt', 'Attentive - Inattentive'),
    ('./nlp/data/vectors/Contemplative_Shallow.pt', 'Shallow - Contemplative'),
    ('./nlp/data/vectors/Evocative_Uninspiring.pt', 'Uninspiring - Evocative'),
    ('./nlp/data/vectors/Witty_Dull.pt', 'Dull - Witty'),
    ('./nlp/data/vectors/Succinct_Rambling.pt', 'Rambling - Succinct'),
    ('./nlp/data/vectors/Arrogant_Humble.pt', 'Humble - Arrogant'),
    ('./nlp/data/vectors/Measured_Impulsive.pt', 'Impulsive - Measured'),
    ('./nlp/data/vectors/Elaborate_Simple.pt', 'Simple - Elaborate'),
    ('./nlp/data/vectors/Unresponsive_Responsive.pt', 'Responsive - Unresponsive'),
    ('./nlp/data/vectors/Courteous_Rude.pt', 'Rude - Courteous'),
    ('./nlp/data/vectors/Tentative_Definite.pt', 'Definite - Tentative'),
    ('./nlp/data/vectors/Compelling_Unconvincing.pt', 'Unconvincing - Compelling'),
    ('./nlp/data/vectors/Casual_Formal.pt', 'Formal - Casual'),
    ('./nlp/data/vectors/Insightful_Superficial.pt', 'Superficial - Insightful'),
    ('./nlp/data/vectors/Assertive_Passive_2.pt', 'Passive - Assertive'),
    ('./nlp/data/vectors/Outgoing_Introverted.pt', 'Introverted - Outgoing'),
    ('./nlp/data/vectors/Concise_Wordy.pt', 'Wordy - Concise'),
    ('./nlp/data/vectors/Confident_Timid.pt', 'Timid - Confident'),
    ('./nlp/data/vectors/Polite_Impolite.pt', 'Impolite - Polite'),
    ('./nlp/data/vectors/Engaging_Aloof.pt', 'Aloof - Engaging')
],
        validators=[DataRequired()],
        render_kw={'class': 'text-options-dropdown'}
    )

    submit = SubmitField('Submit')
