
# Assignment-4
User Interface for ML Models

- Name: Jayant Malik
- Roll no: 242110401
- Online Link: https://spjhfhgzzsshbt5zksvzyw.streamlit.app/ 
- Github Link: https://github.com/devjayantmalik/HumanAIExperienceAssignments

## 1. Target User and Interface Design

### Target User
The primary target users for our absenteeism prediction system are HR professionals and managers in organizations. These users need to:
- Predict potential absenteeism among employees
- Identify factors that contribute to absenteeism
- Make informed decisions about workforce management
- Ensure fair and unbiased treatment of employees

### Interface Design Relation to User Needs
Our interface design is tailored to meet these needs through:

1. **Simple Input Interface**: Users can easily input employee attributes through intuitive form fields in the sidebar.

2. **Clear Visualization of Results**: Predictions are presented through both numerical values and visual representations (gauges and pie charts).

3. **Model Transparency**: The interface provides information about feature importance, helping users understand which factors most influence predictions.

4. **Fairness Indicators**: Performance and fairness metrics are clearly displayed, allowing users to assess the model's fairness across different demographic groups.

5. **Model Limitations**: The interface clearly communicates what the model can and cannot do, helping users make informed decisions.

## 2. Screenshots of Interface

### Main Interface
The main interface consists of:
- A sidebar with input fields for employee attributes
- A main display area showing prediction results
- Sections for model explanation, fairness metrics, and limitations

### Prediction Display
Predictions are shown in two forms:
1. **Regression Prediction**: Shows the predicted absenteeism time in hours with a gauge visualization.
2. **Classification Prediction**: Shows whether an employee is at high or low risk of absenteeism with a probability pie chart.

### Fairness Indicators
The interface displays:
- Model performance metrics (accuracy, precision, recall, F1 score)
- Fairness metrics (statistical parity difference, equal opportunity difference, etc.)
- Visualizations of feature importance

## 3. HAX-Based Evaluation Results

### Clarity (Rating: 4/5)
**Strengths:**
- Clear explanation of model capabilities and limitations
- Intuitive input fields with appropriate labels
- Visual representations of predictions

**Areas for Improvement:**
- Could add more detailed explanations of how the model works internally

### Control (Rating: 5/5)
**Strengths:**
- Users have full control over all input parameters
- Easy to adjust values and see immediate results
- No restrictions on user input

### Feedback (Rating: 4/5)
**Strengths:**
- Immediate visual feedback when inputs are changed
- Clear display of prediction results
- Visual indicators of prediction confidence

**Areas for Improvement:**
- Could add more detailed explanations for each prediction

### Efficiency (Rating: 4/5)
**Strengths:**
- Simple and efficient interface design
- Immediate results without long loading times
- Organized layout with clear sections

**Areas for Improvement:**
- Could optimize loading times for large datasets

### Relevance (Rating: 5/5)
**Strengths:**
- Designed specifically for HR professionals and managers
- Focuses on the key information these users need
- Includes metrics relevant to workforce management decisions

### Trust (Rating: 4/5)
**Strengths:**
- Transparent about model performance and limitations
- Includes fairness metrics to address bias concerns
- Provides information about feature importance

**Areas for Improvement:**
- Could add more transparency about the training data and potential biases

## 4. Conclusion

Our interface successfully addresses the needs of HR professionals and managers by providing a clear, controllable, and trustworthy tool for predicting employee absenteeism. The HAX evaluation shows that our interface performs well across all principles, with particular strengths in control and relevance.

The interface effectively communicates what the model can do, provides clear feedback, and includes important fairness indicators. This helps users make informed decisions while being aware of the model's limitations and potential biases.

Future improvements could focus on enhancing the clarity of model explanations and providing more detailed feedback for predictions. Overall, the interface provides a solid foundation for responsible use of the absenteeism prediction model.
