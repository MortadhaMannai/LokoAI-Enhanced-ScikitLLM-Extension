from loko_extensions.model.components import Component, save_extensions, Arg, Dynamic, Input, Output

model = Arg(name="model", label="OpenAI Model", type="text", helper="Choose one the OpenAI model to use",
            value="gpt-3.5-turbo")
labels = Arg(name="labels", label="Labels", type="text",
             helper="Define the labels, comma separeted, for your classification")
multilabel = Arg(name="multilabel", label="Multilabel Classification", type="boolean",
                 description="Toogle if you want to make multilabel prediction, otherwise multiclass classification will be performed",
                 value=False)

max_labels = Dynamic(name="max_labels", label="Max number of labels", dynamicType="number",
                     description="Max number of labels to assign to an instance", value=3, parent="multilabel",
                     condition="{parent}==true")

clf_args = [model, labels, multilabel, max_labels]
clf_input = [Input(id="data", label="data", service="zero_shot", to="prediction")]
clf_output = [Output(id="prediction", label="prediction")]
clf_comp_description = "### SK-LLM ZeroShot"
llm_clf = Component(name="SK-LLM ZeroShot",inputs=clf_input, outputs=clf_output, description=clf_comp_description,
                    args=clf_args, icon="RiFocus2Fill")

if __name__ == '__main__':
    save_extensions([llm_clf])
