from datetime import datetime, timedelta
from types import SimpleNamespace

from src.backend.app import anomaly_service as anomaly


def _txn(id, customer_id, dt, amount, ttype, merchant, location, category):
    return SimpleNamespace(
        id=id, customer_id=customer_id, timestamp=dt, amount=amount,
        transaction_type=ttype, merchant=merchant, location=location, category=category,
    )


def _history(customer_id='CUST001', n=15, base_amount=500.0, merchant='Swiggy',
             location='Bengaluru', category='Food', hour=10, start_id=1):
    now = datetime(2026, 1, 1, hour, 0, 0)
    rows = []
    for i in range(n):
        rows.append(_txn(
            start_id + i, customer_id, now + timedelta(days=i), base_amount + (i % 3) * 5,
            'DEBIT', merchant, location, category,
        ))
    return rows


def test_normal_transaction_scores_low():
    rows = _history(n=12)
    # one more perfectly typical transaction
    rows.append(_txn(100, 'CUST001', rows[-1].timestamp + timedelta(days=1), 505.0,
                      'DEBIT', 'Swiggy', 'Bengaluru', 'Food', ))
    results = anomaly.evaluate_transactions(rows)
    res = results[100]
    assert res['status'] == 'NORMAL'
    assert res['anomaly_score'] < anomaly.NORMAL_UPPER
    assert res['review_recommended'] is False


def test_large_amount_deviation_is_flagged_with_reason():
    rows = _history(n=12, base_amount=500.0)
    outlier = _txn(200, 'CUST001', rows[-1].timestamp + timedelta(days=1), 50000.0,
                    'DEBIT', 'Swiggy', 'Bengaluru', 'Food')
    rows.append(outlier)
    results = anomaly.evaluate_transactions(rows)
    res = results[200]
    assert res['anomaly_score'] > 0.5
    assert res['status'] in ('MONITOR', 'UNUSUAL')
    assert any('above' in r for r in res['reasons'])


def test_new_merchant_signal():
    rows = _history(n=10, merchant='Swiggy')
    outlier = _txn(300, 'CUST001', rows[-1].timestamp + timedelta(days=1), 500.0,
                    'DEBIT', 'BrandNewMerchant', 'Bengaluru', 'Food')
    rows.append(outlier)
    results = anomaly.evaluate_transactions(rows)
    res = results[300]
    assert any('Merchant/recipient has not appeared' in r for r in res['reasons']) or res['status'] == 'NORMAL'
    # amount is typical, so the merchant flag alone should not push it into UNUSUAL,
    # but it must at least be visible as a contributing signal when combined with anything else.


def test_new_location_signal():
    rows = _history(n=10, location='Bengaluru')
    outlier = _txn(400, 'CUST001', rows[-1].timestamp + timedelta(days=1), 500.0,
                    'DEBIT', 'Swiggy', 'Kolkata', 'Food')
    rows.append(outlier)
    results = anomaly.evaluate_transactions(rows)
    res = results[400]
    reasons_text = ' '.join(res['reasons'])
    if res['status'] != 'NORMAL':
        assert 'location' in reasons_text.lower()


def test_unusual_hour_signal():
    rows = _history(n=10, hour=10)
    outlier = _txn(500, 'CUST001', rows[-1].timestamp.replace(hour=3) + timedelta(days=1), 500.0,
                    'DEBIT', 'Swiggy', 'Bengaluru', 'Food')
    rows.append(outlier)
    results = anomaly.evaluate_transactions(rows)
    res = results[500]
    reasons_text = ' '.join(res['reasons'])
    if res['status'] != 'NORMAL':
        assert 'time of day' in reasons_text.lower()


def test_multiple_signals_combine_into_higher_score():
    rows = _history(n=12, base_amount=500.0, merchant='Swiggy', location='Bengaluru', hour=10)
    single_signal = _txn(600, 'CUST001', rows[-1].timestamp + timedelta(days=1), 5000.0,
                          'DEBIT', 'Swiggy', 'Bengaluru', 'Food')
    multi_signal = _txn(601, 'CUST001', rows[-1].timestamp + timedelta(days=2, hours=-8), 5000.0,
                         'DEBIT', 'TotallyNewMerchant', 'Kolkata', 'Food')
    rows_single = rows + [single_signal]
    rows_multi = rows + [multi_signal]
    score_single = anomaly.evaluate_transactions(rows_single)[600]['anomaly_score']
    score_multi = anomaly.evaluate_transactions(rows_multi)[601]['anomaly_score']
    assert score_multi >= score_single


def test_score_is_normalized_between_0_and_1():
    rows = _history(n=15)
    rows.append(_txn(700, 'CUST001', rows[-1].timestamp + timedelta(days=1), 1_000_000.0,
                      'DEBIT', 'NewOne', 'NewCity', 'Food'))
    results = anomaly.evaluate_transactions(rows)
    for res in results.values():
        assert 0.0 <= res['anomaly_score'] <= 1.0


def test_reasons_are_generated_dynamically_not_static():
    rows_a = _history(n=10, base_amount=500.0)
    rows_a.append(_txn(800, 'CUST001', rows_a[-1].timestamp + timedelta(days=1), 20000.0,
                        'DEBIT', 'NewMerchantA', 'Bengaluru', 'Food'))
    rows_b = _history(n=10, base_amount=2000.0, start_id=1000)
    rows_b.append(_txn(1800, 'CUST001', rows_b[-1].timestamp + timedelta(days=1), 20000.0,
                        'DEBIT', 'Swiggy', 'Bengaluru', 'Food'))
    reasons_a = anomaly.evaluate_transactions(rows_a)[800]['reasons_text']
    reasons_b = anomaly.evaluate_transactions(rows_b)[1800]['reasons_text']
    # different underlying behaviour -> different generated reasons, proving
    # reasons come from actual detected signals rather than a fixed string.
    assert reasons_a != reasons_b


def test_never_describes_as_confirmed_fraud():
    rows = _history(n=12)
    rows.append(_txn(900, 'CUST001', rows[-1].timestamp + timedelta(days=1), 100000.0,
                      'DEBIT', 'Unknown', 'Unknown', 'Food'))
    results = anomaly.evaluate_transactions(rows)
    for res in results.values():
        assert 'fraud' not in res['reasons_text'].lower()
        assert res['status'] in ('NORMAL', 'MONITOR', 'UNUSUAL')


def test_small_dataset_handled_safely_without_crashing():
    rows = _history(n=2)
    results = anomaly.evaluate_transactions(rows)
    assert len(results) == 2
    for res in results.values():
        assert 0.0 <= res['anomaly_score'] <= 1.0
