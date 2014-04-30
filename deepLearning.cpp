#include <time.h>

#include <iostream>
#include <vector>
#include <fstream>

using namespace std;


const double Eta     = 0.5;
const double Alpha   = 0.8;
const double PAI     = 3.14159265359;
const double ErrorEv = 0.08;
const double A       = 1.0;
const double Lambda  = PAI;

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

    // ���͑w�̓��͐M���͂��̂܂ܓ����
    void setX(double x) { this->x = x; }
};

void forwardPropagation(vector<vector<Neuron*>>& neurons, const vector<double>& inp)
{
    for(int i = 0; i < neurons[0].size(); i++) {
        neurons[0][i]->setX(inp[i]);
    }
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
        cout << "now_delta:";
        for(int j = 0; j < neurons[i].size(); j++) {
            cout << now_delta[j] << ", ";
        }
        cout << endl;

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

    // �j���[������3�w��1-2-1�ɂ���
    neurons.resize(3);
    neurons[0].resize(1);
    neurons[1].resize(2);
    neurons[2].resize(1);

    vector<double> w(1);
    w[0] = 1.0;
    neurons[0][0] = new Neuron(w, 0.0);

    w = vector<double>(1);
    w[0] = 0.5;
    neurons[1][0] = new Neuron(w, 0.0);
    w[0] = 0.7;
    neurons[1][1] = new Neuron(w, 0.0);
    w = vector<double>(2);
    w[0] = 0.1;
    w[1] = 0.3;
    neurons[2][0] = new Neuron(w, 0.0);

    /*
    // �j���[�����l�b�g���[�N���\�z����
    for(int i = 1; i < neurons.size(); i++) {
        w = vector<double>(neurons[i - 1].size());
        for(int j = 0; j < neurons[i].size(); j++) {
            for(int k = 0; k < neurons[i - 1].size(); k++) {
                w[k] = my_rand(-10, 10, 2);
            }
            neurons[i][j] = new Neuron(w, my_rand(-1, 1, 2));
        }
    }
    */


    // ���t�f�[�^�̍쐬
    const int Patterns = 200;
    vector<double> inp_dats[Patterns];
    vector<double> tsignal[Patterns];

    for(int i = 0; i < Patterns; i++) {
        for(int j = 0; j < neurons[0].size(); j++) {
            inp_dats[i].push_back(my_rand(-1, 1, 2));
        }
        for(int j = 0; j < neurons[neurons.size() - 1].size(); j++) {
            tsignal[i].push_back((A * sin(Lambda * inp_dats[i][j]) + A) / (2 * A));
        }
    }
    
    // �w�K������
    double vError = ErrorEv + 1.0;
    for(int i = 0; vError > ErrorEv && i < 0; i++) {
        for(int j = 0; j < Patterns; j++) {
            forwardPropagation(neurons, inp_dats[j]);
            backPropagation(neurons, tsignal[j]);
        }
    }


    inp_dats[0][0] = 0.3;
    tsignal[0][0]  = 0.0164;
    forwardPropagation(neurons, inp_dats[0]);
    backPropagation(neurons, tsignal[0]);


    // �j���[�����f�[�^�̏o��
    for(int i = 0; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            cout << "neurons[" << i << "][" << j << "]:"
                << neurons[i][j]->getW().size() << ", " << neurons[i][j]->getBias() << ", "
                << neurons[i][j]->getU() << ", " << neurons[i][j]->getX() << endl;
      
            for each(double w in neurons[i][j]->getW()) {
                cout << w << ", ";
            }
            cout << endl;
            for each(double delta_w in neurons[i][j]->getDeltaW()) {
                cout << delta_w << ", ";
            }
            cout << endl;
        }
    }

    // ���ʂ��o��
    ofstream ofs_sin("out_sin.dat");
    ofs_sin << "# ";
    for(int i = 0; i < inp_dats[0].size(); i++) {
        ofs_sin << "inp_dats[" << i << "]" << "\t";
    }
    for(int i = 0; i < tsignal[0].size(); i++) {
        ofs_sin << "tsignal[" << i << "]" << "\t";
    }
    for(int i = 0; i < neurons[neurons.size() - 1].size(); i++) {
        ofs_sin << "output[" << i << "]" << "\t";
    }
    ofs_sin << endl;

    for(int i = 0; i < Patterns; i++) {
        for(int j = 0; j < inp_dats[i].size(); j++) {
            ofs_sin << inp_dats[i][j] << "\t";
        }
        forwardPropagation(neurons, inp_dats[i]);

        for(int j = 0; j < tsignal[i].size(); j++) {
            ofs_sin << (2 * A * tsignal[i][j] - A) << "\t";
        }
        for(int j = 0; j < neurons[neurons.size() - 1].size(); j++) {
            ofs_sin << (2 * A * neurons[neurons.size() - 1][j]->getX() - A) << "\t";
        }
        ofs_sin << endl;
    }

    // �j���[�����̍폜�i���I�Ɋm�ۂ������߁j
    for(int i = 0; i < neurons.size(); i++) {
        for(int j = 0; j < neurons[i].size(); j++) {
            delete neurons[i][j];
        }
    }  

    return 0;
}