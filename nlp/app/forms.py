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
        choices=[
            ('option1', 'Option 1 Text'),
            ('option2', 'Option 2 Text'),
            ('option3', 'Option 3 Text'),
        ],
        validators=[DataRequired()],
        render_kw={'class': 'text-options-dropdown'}
    )

    submit = SubmitField('Submit')
