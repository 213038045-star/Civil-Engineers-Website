from flask import Blueprint, render_template

projects_bp = Blueprint('projects', __name__)

@projects_bp.route('/projects')
def projects():
    return render_template('design/projects/projects.html')


footing_bp = Blueprint('footing', __name__)
@footing_bp.route('/projects/footing_design')
def footing_design():
    return render_template('design/projects/footing_design.html')


@footing_bp.route('/projects/footing_design')
def footing_design():
    return render_template('design/projects/mat or raft_design.html')



@footing_bp.route('/projects/pavement_design')
def pavement_design():
    return render_template('design/projects/pavement_design_tool.html')