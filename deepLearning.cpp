#include <time.h>

#include <iostream>
#include <vector>

using namespace std;


const double Eta   = 0.5;
const double Alpha = 0.8;

inline double fout(double x)
{
    return 1 / (1 + exp(-x));
}

inline double delta_fout(double fout_value)
{
    return fout_value * (1.0 - fout_value);
}

// dec_point_num <= 3�������B4���班�����삪��
double my_rand(double min, double max, int dec_point_num)
{
    double value = 1.0;
    for(int i = 0; i < dec_point_num; i++){
        value *= 10.0;
    }
    return (double)(rand() % (int)((max - min) * value + 1)) / value + min;
}

class Neuron {
    double u;
    double x;
    vector<double> w;
    vector<double> delta_w;
    double bias;
    double delta_bias;

public:
    Neuron(const vector<double>& w, double bias)
        : w(w), bias(bias), u(0.0), x(0.0), delta_w(0), delta_bias(0.0)
    {
        delta_w.resize(w.size());
        for(int i = 0; i < delta_w.size(); i++) {
            delta_w[i] = 0.0;
        }
    }

    // --- getter --- //
    double getU() const { return u; }
    double getX() const { return x; }
    const vector<double>& getW() const { return w; }
    const vector<double>& getDeltaW() const { return delta_w; }
    double getBias() const { return bias; }
    double getDeltaBias() const { return delta_bias; }

    // --- setter --- //
    void setW(const vector<double>& w) { this->w = w; }
    void setDeltaW(const vector<double>& delta_w) { this->delta_w = delta_w; }
    void setBias(double bias) { this->bias = bias; }
    void setDeltaBias(double delta_bias) { this->delta_bias = delta_bias; }

    void scalarProduct(const vector<double>& in) {
        if(w.size() == in.size()) {
            u = 0.0;
            for(int i = 0; i < w.size(); i++) {
                u += w[i] * in[i];
            }
            x = fout(u + bias);
        }
    }
};

void forwardPropagation(vector<vector<Neuron*>>& neurons, const vector<double>& inp)
{
    neurons[0][0]->scalarProduct(inp);
    for(int i = 1; i < neurons.size(); i++) {
        vector<double> inp(neurons[i - 1].size());
        for(int j = 0; j < inp.size(); j++) {
            inp[j] = neurons[i - 1][j]->getX();
        }
        for(int j = 0; j < neurons[i].size(); j++) {
            neurons[i][j]->scalarProduct(inp);
        }
    }
}

void backPropagation(vector<vector<Neuron*>>& neurons, const vector<double>& tsignal)
{

    vector<Neuron*>& out_neurons = neurons[neurons.size() - 1];     // �o�͑w�j���[����
    // ���͐M���Ƌ��t�M���Ƃ̌덷�����߂�
    vector<double> dwo(out_neurons.size());
    for(int i = 0; i < dwo.size(); i++) {
        dwo[i] = (tsignal[i] - out_neurons[i]->getX()) * delta_fout(out_neurons[i]->getX());
    }


    // �o�͑w�̌����׏d�l��ύX����
    for(int i = 0; i < out_neurons.size(); i++) {
        vector<double> w       = out_neurons[i]->getW();
        vector<double> delta_w = out_neurons[i]->getDeltaW();
        for(int j = 0; j < w.size(); j++) {
            delta_w[j] = Eta * dwo[i] * neurons[neurons.size() - 2][j]->getX() + Alpha * delta_w[j];
            w[j] += delta_w[j];
        }
        out_neurons[i]->setW(w);
        out_neurons[i]->setDeltaW(delta_w);
    }

    // �o�͑w�̂������l��ύX����
    for(int i = 0; i < out_neurons.size(); i++) {
        double bias = out_neurons[i]->getBias();
        double delta_bias = out_neurons[i]->getDeltaBias();
        delta_bias = Eta * dwo[i] + Alpha * delta_bias;
        bias += delta_bias;
        out_neurons[i]->setBias(bias);
        out_neurons[i]->setDeltaBias(delta_bias);
    }

    // ���ԑw�̌����׏d�l�Ƃ������l��ύX����
    vector<double> up_delta  = dwo;
    for(int i = (int)neurons.size() - 2; i >= 1; i--) {
        vector<double> now_delta(neurons[i].size());

        // �덷�`���Ō덷�����߂�
        for(int j = 0; j < neurons[i].size(); j++) {
            double sum = 0.0;
            
            for(int k = 0; k < neurons[i + 1].size(); k++) {
                sum += up_delta[k] * neurons[i + 1][k]->getW()[j];
            }
            now_delta[j] = delta_fout(neurons[i][j]->getX()) * sum;
        }

        // ���݂̑w�̌����׏d�l��ύX����
        for(int j = 0; j < neurons[i].size(); j++) {
            vector<double> w = neurons[i][j]->getW();
            vector<double> delta_w = neurons[i][j]->getDeltaW();
            for(int k = 0; k < w.size(); k++) {
                delta_w[k] = Eta * now_delta[j] + neurons[i - 1][k]->getX() + Alpha * delta_w[k];
                w[k] += delta_w[k];
            }
            neurons[i][j]->setW(w);
            neurons[i][j]->setDeltaW(delta_w);
        }

        // ���݂̑w�̂������l��ύX����
        for(int j = 0; j < neurons[i].size(); j++) {
            double bias = neurons[i][j]->getBias();
            double delta_bias = neurons[i][j]->getDeltaBias();
            delta_bias = Eta * now_delta[j] + Alpha * delta_bias;
            bias += delta_bias;
            neurons[i][j]->setBias(bias);
            neurons[i][j]->setDeltaBias(delta_bias);
        }

        up_delta = now_delta;
    }
}

int main()
{
    srand((unsigned int)time(NULL));

    vector<vector<Neuron*>> neurons(0, vector<Neuron*>(0));

    neurons.resize(5);
    neurons[0].resize(1);
    neurons[1].resize(5);
    neurons[2].resize(3);
    neurons[3].resize(5);
    neurons[4].resize(1);

    vector<double> w(1);
    w[0] = 1.0;
    neurons[0][0] = new Neuron(w, 0.0);

    for(int i = 1; i < neurons.size(); i++) {
        w = vector<double>(neurons[i - 1].size());
        for(int j = 0; j < neurons[i].size(); j++) {
            for(int k = 0; k < neurons[i - 1].size(); k++) {
                w[k] = my_rand(-1, 1, 2);
            }
            neurons[i][j] = new Neuron(w, my_rand(-1, 1, 2));
        }
    }

    vector<double> inp(1);
    inp[0] = my_rand(-1, 1, 2);
    forwardPropagation(neurons, inp);

    vector<double> tsignal(1);
    tsignal[0] = my_rand(0, 1, 2);
    backPropagation(neurons, tsignal);


    for(int i = 0; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            cout << "neurons[" << i << "][" << j << "]:"
                << neurons[i][j]->getW().size() << ", " << neurons[i][j]->getBias() << ", "
                << neurons[i][j]->getU() << ", " << neurons[i][j]->getX() << endl;
      
            for each(double w in neurons[i][j]->getW()) {
                cout << w << ", ";
            }
            cout << endl;
        }
    }

    for(int i = 0; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            delete neurons[i][j];
        }
    }  

    return 0;
}