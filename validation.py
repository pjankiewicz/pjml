import hashlib
import os
import pickle

from types import FunctionType


from sklearn.cross_validation import KFold


def md5(s):
    return hashlib.md5(s).hexdigest()


class Prediction(object):

    def __init__(self, model_cls, dataset, save=True):
        self.dataset = dataset
        self.model_cls = model_cls
        self.save = save
        self.variance_expansion = variance_expansion

    def _load_data(self):
        X_tr, y_tr, X_te = self.dataset.load_data()
        X_train = np.array(X_tr)
        y_train = np.array(y_tr, dtype=int)
        X_test = np.array(X_te)
        return X_train, y_train, X_test

    def _extract_model_parameters(self):
        if hasattr(self.model_cls, "estimator_params"):
            return dict((param, getattr(self.model_cls, param)) for param in
                        self.model_cls.estimator_params)
        elif hasattr(self.model_cls, "_get_param_names"):
            return dict((param, getattr(self.model_cls, param)) for param in
                        self.model_cls._get_param_names())
        else:
            raise NotImplemented, "model_cls must implement estimator_params" \
                                  " or _get_param_names()"

    def _get_model_name(self):
        return self.model_cls.__class__.__name__

    def _hash_params(self):
        params = self._extract_model_parameters()
        params_str = " ".join(["%s=%s" % (k, v) for k, v in params.items()])
        return md5(params_str)

    def _hash_function_kwargs(self, func_kwargs):
        ret = []
        for k, v in func_kwargs.iteritems():
            if type(v) == FunctionType:
                ret.append(str(k) + '_' + v.__name__)
            elif hasattr(v, '__name__'):
                ret.append(str(k) + '_' + v.__name__ + '_' + str(v))
            elif hasattr(v.__class__, '__name__'):
                ret.append(str(k) + '_' + v.__class__.__name__ + '_' + str(v))
            else:
                ret.append(str(k) + '_' + str(v))

        if len(ret) == 0:
            return ''
        else:
            return md5(''.join(ret)) + '_'

    def _generate_filename(self):
        filename = "%s_%s%s_%s.pkl" % (self.dataset.func.__name__,
                                       self._hash_function_kwargs(
                                           self.dataset.kwargs),
                                       type(self.model_cls).__name__,
                                       self._hash_params())
        return os.path.join(PREDICTION_PATH, filename)

    def _expand_dataset(self, X, y, variance):
        X_rows = []
        y_ = []
        X = np.array(X)
        y = np.array(y)
        for n in range(X.shape[0]):
            ratings = variance_lookup.reverse(variance[n], y[n])
            for rating in ratings:
                X_rows.append(X[n, :])
                y_.append(rating)
        print len(X_rows)
        return np.vstack(X_rows), np.array(y_)

    def cross_validate(self):
        if self.save is True:
            save_path = self._generate_filename()
            if os.path.exists(save_path):
                print "model %s exists skipping..." % save_path
                return

            # reserve the filename
            touch(save_path)

        X_train, y_train, X_test = self._load_data()
        data_df = load_data_as_df()
        variance = np.array(data_df["relevance_variance"])
        train_n = X_train.shape[0]
        ld_n = X_test.shape[0]
        all_n = train_n + ld_n

        cv = KFold(n=X_train.shape[0], n_folds=10, random_state=12345)

        # save raw predictions in a dataframe
        predictions = pd.DataFrame({
            "obs_type": ["train"]*train_n + ["leaderboard"]*ld_n,
            "prediction": np.zeros(all_n, dtype=np.float32)
        })

        cv_qwk = []

        oof_predictions_tr = np.zeros(shape=(train_n,))
        for fold, (tr_idx, te_idx) in enumerate(cv):

            if self.variance_expansion:
                X_train_, y_train_ = self._expand_dataset(X_train[tr_idx],
                                                          y_train[tr_idx],
                                                          variance[tr_idx])
            else:
                X_train_ = X_train[tr_idx]
                y_train_ = y_train[tr_idx]

            self.model_cls.fit(X_train_, y_train_)
            preds = self.model_cls.predict(X_train[te_idx])
            oof_predictions_tr[te_idx] = preds

            # show some info about the cv error
            preds_ = grade_on_a_curve(preds, y_train[tr_idx])
            qwk_fold = quadratic_weighted_kappa(preds_, y_train[te_idx])
            cv_qwk.append(qwk_fold)
            print "fold %d: %.4f" % (fold + 1, qwk_fold)

        # show final cv qwk
        oof_predictions_tr_ = grade_on_a_curve(oof_predictions_tr, y_train)
        qwk = quadratic_weighted_kappa(oof_predictions_tr_, y_train)
        print "Final: %.4f" % (qwk)

        if self.variance_expansion:
            X_train_, y_train_ = self._expand_dataset(X_train,
                                                      y_train,
                                                      variance)
        else:
            X_train_ = X_train
            y_train_ = y_train

        self.model_cls.fit(X_train_, y_train_)
        predictions_ld = self.model_cls.predict(X_test)

        predictions["prediction"] = np.concatenate(
            [oof_predictions_tr, predictions_ld])

        to_save = {
            "predictions": predictions,
            "cv_test_errors": qwk,
            "oof_test_error": cv_qwk,
            "model_cls_params": self._extract_model_parameters(),
            "model_cls_name": self._get_model_name(),
            'dataset_params': self.dataset.kwargs
        }

        if self.save is True:
            pickle.dump(to_save, open(save_path, "wb"))
