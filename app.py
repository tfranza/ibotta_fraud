# https://www.youtube.com/watch?v=qNF1HqBvpGE
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from joblib import load

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def home_page():
	request_type_str = request.method
	if request_type_str == 'GET': 
		return render_template('index.html', href = 'static/anomaly.png')
	else: 
		text = request.form['text']
		path = "static/predictions_pic.png"
		#Make the graph here 
		model = load('model.joblib')
		np_array = floats_string_to_np_arr(text)
		make_picture('AgesAndHeights.pkl', model, np_array, path)
		return render_template('index.html', href =path)


@app.route("/<accountID>")
def clustered_accounts(accountID):
	return f"the account you entered is {accountID}"

def make_picture(training_data_filename, model, new_inp_np_arr, output_file):
  data = pd.read_pickle(training_data_filename)
  ages = data['Age']
  data = data[ages > 0]
  ages = data['Age']
  heights = data['Height']
  x_new = np.array(list(range(19))).reshape(19, 1)
  preds = model.predict(x_new)

  fig = px.scatter(x=ages, y=heights, title="Height vs Age of People", labels={'x': 'Age (years)',
                                                                                'y': 'Height (inches)'})

  fig.add_trace(go.Scatter(x=x_new.reshape(19), y=preds, mode='lines', name='Model'))

  new_preds = model.predict(new_inp_np_arr)

  fig.add_trace(go.Scatter(x=new_inp_np_arr.reshape(len(new_inp_np_arr)), y=new_preds, name='New Outputs', mode='markers', marker=dict(color='purple', size=20, line=dict(color='purple', width=2))))
  
  fig.write_image(output_file, width=800)#, engine='kaleido')
  #fig.show()


def floats_string_to_np_arr(floats_str):
  def is_float(s):
    try:
      float(s)
      return True
    except:
      return False
  floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
  return floats.reshape(len(floats), 1)



if __name__ == "__main__":
	app.run(debug=True, host='0.0.0.0', port=5000)