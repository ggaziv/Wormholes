from wormholes import PROJECT_ROOT
from wormholes.tools import *
from wormholes.tools.label_maps import CLASS_DICT
from matplotlib import pyplot as plt
import scipy
import pandas as pd
import seaborn as sns
plt.rcParams.update({'font.size': 14})
sns.set_context('talk')


class_dict = dict(list(CLASS_DICT['RestrictedImageNet'].items())[1:])
class_dict_inv = {v: k for k, v in class_dict.items()}


def check_uniform_sample_count(df2, hue='subject', x='budget', var=None):
    """Check same number of samples per data point
    """
    samples_count = {s: len(df2[df2[hue] == s].dropna(subset=var))//len(df2[df2[hue] == s].dropna(subset=var)[x].unique()) 
                        for s in df2[hue].unique() if len(df2[df2[hue] == s].dropna(subset=var)) > 0}
    samples_count_unique = np.unique(list(samples_count.values()))
    if len(samples_count_unique) == 1:
        cprint(f"(n={samples_count_unique[0]})", 'yellow')
    else:
        cprint(f"Warning [non-uniform sample count]: {samples_count}", 'red')
    
        
class Human_Model_PlotObject:
    def __init__(self, df, hue='case'):
        df1 = df.copy()
        self.hue = hue
        df1['is_control'] = df1.case.isin(['No perturb.', 'Catch'])
        df1 = df1.dropna(subset='budget')

        df1[self.hue] = df1[self.hue].str.replace('_', ' ')
        self.df = df1.reset_index()

    @staticmethod
    def add_inferred_catch(df1, attacker_field='case', var_list=['prob_choose_target']):
        # Infer Catch from No perturb.
        df1 = pd.concat([df1, 
                        df1[df1[attacker_field] == 'No perturb.'].assign(**{
                            var: lambda x: 1 - x[var] for var in var_list} | {
                            attacker_field: 'Catch'
                            })])
        return df1
    
    @staticmethod
    def replicate_control_at_high_x(df1, x='budget'):
        # Replicate the control data also at high x-value to form a line
        df1 = pd.concat([df1, df1[df1.is_control].assign(**{x: df1[x].max()})]).reset_index()
        return df1
    
    
class Human_Model_2AFC(Human_Model_PlotObject):
    def plot(self, x='budget', 
             var_list=['mean_prob_choose_target', 
                       'pred_prob_choose_target'],
             add_inferred_catch=True,
             custom_palette=None,
             style=None,
             xscale='log'):
        df1 = self.df
        if isinstance(var_list, list):
            var_dict = {k: v for k, v in zip(*([var_list] * 2))}
        else:
            var_dict = var_list
        if add_inferred_catch:
            df1 = self.add_inferred_catch(df1, 'case', var_dict.keys())
        df1 = self.replicate_control_at_high_x(df1, x)
        
        fig = plt.figure().add_gridspec(1, len(var_dict), wspace=.3).subplots(sharey=False)
        for i, (var, var_name) in enumerate(var_dict.items()):
            plt.subplot(1, len(var_dict), i + 1)
            df2 = df1.dropna(subset=var)
            check_uniform_sample_count(df2, self.hue, x, var)
            g = lineplot_two_stage(data=df2,
                                   x=x,
                                   y=var,
                                   hue=self.hue,
                                   pal=custom_palette,
                                   alpha=.7,
                                   style_control='is_control',
                                   style=style,
                                   legend=i==len(var_dict)-1, 
                                   )
            plt.grid(True)
            if x == 'budget':
                xlabel = r"Pixel budget allowable ($\epsilon^{224}_{Madry}$)"
            elif x in ['dx', 'dx_binned']:
                xlabel = r"Pixel modulation $\ell_{2}$-norm"
            elif x == 'class_name':
                xlabel = 'Class'
            plt.xlabel(xlabel)
            plt.ylabel(var_name.capitalize().replace('_', ' '))
            plt.xscale(xscale)
            plt.ylim(-.03, 1.03)

        plt.suptitle(f'2AFC Class Modulation', weight='bold', fontsize=20)
        make_legend(g, bbox_to_anchor=(1.05, 1.1))
        plt.gcf().set_size_inches(12, 4)
        return fig


class AlignmentPlot:
    def __init__(self, df, x='budget', attacker_field='case', model_var='pred_prob_choose_target', 
                 add_inferred_catch=True):
        self.x = x
        self.attacker_field = attacker_field
        id_vars = [self.x, self.attacker_field]
        df1 = self.make_df(df, model_var, id_vars)
        df1['is_control'] = df1[self.attacker_field].isin(['No perturb.', 'Catch'])
        if add_inferred_catch:
            df1 = Human_Model_PlotObject.add_inferred_catch(df1, attacker_field, var_list=['prob_choose_target'])
        
        # Add control cases per plot
        keys = ['Catch', 'No perturb.']
        for k in keys:
            df2 = df1[df1[self.attacker_field] == k]
            if df2.empty: 
                continue
            df2 = df2.assign(subject=df2.subject.apply(lambda x: f"{x} {k}"))
            for k1 in df1[self.attacker_field].unique():
                if k1 not in keys:
                    df2[self.attacker_field] = k1
                    df1 = pd.concat([df1[df1[self.attacker_field] != k], df2])
        self.df = df1
    
    def make_df(self, df, model_var, id_vars):
        if isinstance(model_var, dict):
            df1 = (
                df.rename(columns=model_var)
                .melt(id_vars=id_vars, value_vars=list(model_var.values()), var_name='subject', value_name='prob_choose_target')
                ).copy()
        else:    
            df1 = (
                df.rename(columns={model_var: 'Model'})
                .melt(id_vars=id_vars, value_vars=['Model'], var_name='subject', value_name='prob_choose_target')
            ).copy()
        return df1
    
    def plot(self, concise=True, 
             ylabel="Prob. choose target", 
             **kwargs):
        return self._plot(self.df, self.attacker_field, self.x, concise, 
                          ylabel=ylabel, **kwargs)
        
    @staticmethod
    def _plot(df1, attacker_field='case', x='budget', concise=True, 
              ylabel="Prob. choose target", bbox_to_anchor=(1.45, .7),
              style=None, custom_palette=None, facet_kws={}, grid=None):
        """Assumes controls are already added to df1
        """
        df1 = Human_Model_PlotObject.replicate_control_at_high_x(df1, x)        
        
        df1[attacker_field] = df1[attacker_field].astype(str).apply(lambda x: x.replace('_', ' ').title().replace('Rin', 'RIN'))
        if concise:
            df1 = df1[df1[attacker_field].isin(['Resnet50 Robust Mapped RIN', 'Resnet50 Vanilla Mapped RIN'])]
        # Remove Catch for model
        df1 = df1[(~df1.subject.str.contains('Catch')) | (df1.subject.str.contains('Human Catch'))]
        # Remove No Perturb. for model
        df1 = df1[(~df1.subject.str.contains('No perturb.')) | (df1.subject.str.contains('Human No perturb.'))]
        
        AlignmentPlot.check_uniform_sample_count(df1, attacker_field)
        
        def my_lineplot(*args, data=None, pal=None, **kwargs):     
            if custom_palette is None:
                subject_names = list(data.subject.unique())
                pal = sns.color_palette('bright', len(subject_names))
                if 'Human' in subject_names:
                    pal.insert(subject_names.index('Human'), (0,0,0))
            else:
                pal = custom_palette
            return lineplot_two_stage(*args, **kwargs, data=data, pal=pal)
            
        g = sns.FacetGrid(df1, row=attacker_field, sharex=False, **facet_kws)
        g.map_dataframe(my_lineplot, 
                        x=x, y='prob_choose_target', 
                        hue='subject', 
                        alpha=.7,
                        style_control='is_control',
                        style=style,
                    )  

        if x == 'budget':
            xlabel = r"Pixel budget allowable ($\epsilon^{224}_{Madry}$)"
        elif x in ['dx', 'dx_binned']:
            xlabel = r"Pixel modulation $\ell_{2}$-norm"
        g.set(xlabel=xlabel, ylabel=ylabel, xscale='log')
        g.set_titles('{row_name}')
        g.add_legend()
        make_legend(g, bbox_to_anchor=bbox_to_anchor)
        if grid is not None:
            for ax in g.axes.flat:
                if grid == 'y':
                    ax.grid(axis='y')
                else:
                    ax.grid(True)

        plt.gcf().set_size_inches(7, 4 * len(df1[attacker_field].unique()))
        plt.tight_layout()
        return g

    @staticmethod
    def check_uniform_sample_count(df1, attacker_field, hue='subject', x='budget'):
        """Check same number of samples per data point
        """
        for c in df1[attacker_field].unique():
            df2 = df1[df1[attacker_field] == c]
            check_uniform_sample_count(df2, hue, x)
        

class Model_Human_Alignment(AlignmentPlot):
    def __init__(self, *args, **kwargs):
        self.human_field = kwargs.pop('human_field', 'mean_prob_choose_target')
        super().__init__(*args, **kwargs)
        
    def make_df(self, df, model_var, id_vars):
        if isinstance(model_var, dict):
            df1 = (
                df.rename(columns={self.human_field: 'Human'} | model_var)
                .melt(id_vars=id_vars, value_vars=['Human'] + list(model_var.values()), var_name='subject', value_name='prob_choose_target')
                ).copy()
        else:    
            df1 = (
                df.rename(columns={self.human_field: 'Human'} | {model_var: 'Model'})
                .melt(id_vars=id_vars, value_vars=['Human', 'Model'], var_name='subject', value_name='prob_choose_target')
            ).copy()
            
        return df1
    
    
    def plot(self, concise=True, 
             ylabel="Prob. choose target", **kwargs):
        return self._plot(self.df, self.attacker_field, self.x, concise, 
                          ylabel=ylabel, **kwargs)


class Model_Human_Alignment_MultiWay:
    def __init__(self, df, x='class_name', col='budget', attacker_field='case', model_var='pred_prob_choose', 
                 add_inferred_catch=False):    
        self.x = x
        self.col = col
        self.attacker_field = attacker_field
        id_vars = [self.x, self.col, self.attacker_field]
        if isinstance(model_var, dict):
            df1 = (
                df.rename(columns={'mean_prob_choose': 'Human'} | model_var)
                .melt(id_vars=id_vars, value_vars=['Human'] + list(model_var.values()), var_name='subject', value_name='prob_choose')
                ).copy()
        else:    
            df1 = (
                df.rename(columns={'mean_prob_choose': 'Human', model_var: 'Model'})
                .melt(id_vars=id_vars, value_vars=['Human', 'Model'], var_name='subject', value_name='prob_choose')
            ).copy()
        df1 = df1.dropna(subset='prob_choose')
        df1['is_control'] = df1[self.attacker_field].isin(['No perturb.', 'Catch'])

        # Add control cases per plot
        keys = ['Catch', 'No perturb.']
        for k in keys:
            df2 = df1[df1[self.attacker_field] == k]
            df2 = df2.assign(subject=df2.subject.apply(lambda x: f"{x} {k}"))
            for k1 in df1[self.attacker_field].unique():
                if k1 not in keys:
                    df2[self.attacker_field] = k1
                    df1 = pd.concat([df1[df1[self.attacker_field] != k], df2])
        self.df = df1

    
    def plot(self, concise=True, bbox_to_anchor=(1.45, .7), polar=False):
        df1 = self.df
        df1[self.attacker_field] = df1[self.attacker_field].apply(lambda x: x.replace('_', ' ').title().replace('Rin', 'RIN'))
        if concise:
            df1 = df1[df1[self.attacker_field].isin(['Resnet50 Robust Mapped RIN', 'Resnet50 Vanilla Mapped RIN'])]
        # Remove Catch for model
        df1 = df1[(~df1.subject.str.contains('Catch')) | (df1.subject.str.contains('Human Catch'))]
        # Remove No Perturb. for model
        df1 = df1[(~df1.subject.str.contains('No perturb.')) | (df1.subject.str.contains('Human No perturb.'))]
        if polar:
            return self.plot_polar(df1, bbox_to_anchor)
        else:
            return self.plot_cartesian(df1, bbox_to_anchor)

     
    def plot_cartesian(self, df1, bbox_to_anchor):
        for row_name in natsorted(df1[self.attacker_field].unique()):
            df2 = df1[df1[self.attacker_field] == row_name]
            g = sns.FacetGrid(df2, col=self.col, sharex=False)
            g.map_dataframe(sns.lineplot, 
                            x=self.x, 
                            y='prob_choose', 
                            hue='subject', 
                            palette=sns.color_palette('bright', len(df2.subject.unique())),
                            alpha=.7,
                            err_kws=dict(alpha=.1),
                            style='is_control',
                            size='is_control', sizes=(.75, 3.5),
                            markers=True
                        )
            g.set_xticklabels(rotation=90)
            if self.col == 'budget':
                col_label = 'budget'
            elif self.col in ['dx', 'dx_binned']:
                col_label = 'Budget used'
            elif self.col == 'target_class_name':
                col_label = 'target'
            g.set(xlabel=self.x.replace('_', ' ').title(), ylabel="Prob. choose")
            plt.suptitle(row_name, weight='bold')
            g.set_titles(col_label + ' [{col_name}]')
            g.add_legend()
            make_legend(g, bbox_to_anchor=bbox_to_anchor)
            for ax in g.axes.flat:
                ax.grid(axis='y')

            plt.gcf().set_size_inches(3 * len(df2[self.col].unique()), 4)
            plt.tight_layout()
            plt.show()
        
        
class LapseRateNorm:
    @staticmethod
    def compute_gamma(pcatch, N=9):
        return (1 - pcatch) / (1 - 1/N)
    
    @staticmethod
    def apply_gamma(p_choose_target, gamma, N=9):
        return (p_choose_target - gamma / N) / (1 - gamma)
    
    
def lineplot_two_stage(*args, data=None, pal=None, **kwargs):        
    hue = kwargs['hue']
    style_control = kwargs.pop('style_control')
    data1 = data[~data[style_control]]
    pal1 = pal
    is_list_palette = not pal is None and not isinstance(pal, dict)
    if is_list_palette:
        pal1 = pal[:len(data1[hue].unique())]
    kwargs_style = dict(style=kwargs.pop('style', None))
    if kwargs_style['style'] is None:
        kwargs_style['marker'] = 'o'
    else:
        kwargs_style['markers'] = True
    ax = sns.lineplot(*args, data=data1, **kwargs, 
                        palette=pal1,
                        linewidth=3.5,
                        err_style='bars',
                        err_kws=dict(capsize=3, capthick=2, elinewidth=2),
                        **kwargs_style)
    data1 = data[data[style_control]]
    if is_list_palette:
        pal1 = pal[-len(data1[hue].unique()):]
    ax = sns.lineplot(*args, data=data1, **kwargs, 
                        palette=pal1,
                        linewidth=.75,
                        linestyle='--',
                        err_kws=dict(alpha=.1),
                        ax=ax)
    return ax


def make_legend(g, bbox_to_anchor=(1.05, .6), legend_title=None):
    if isinstance(g, sns.FacetGrid):
        leg = g._legend
    else:
        leg = g.get_legend()
    leg.set_bbox_to_anchor(bbox_to_anchor)
    leg.set_title(legend_title)
    leg.get_frame().set_linewidth(0.0)